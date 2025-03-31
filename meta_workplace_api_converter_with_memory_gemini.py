import pandas as pd
import json
import os
import warnings
import glob
from tqdm import tqdm
import gc  # Import garbage collector
import pyarrow as pa
import pyarrow.parquet as pq

warnings.filterwarnings("ignore")

# --- Helper Function to Optimize Data Types (Simplified) ---
# Keeping the simplified version from the previous step as aggressive downcasting caused errors
def optimize_dtypes(df):
    if df.empty: return df
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == 'object':
            try:
                num_unique_values = len(df[col].unique())
                num_total_values = len(df[col])
                if num_unique_values / num_total_values < 0.5 or num_unique_values < 100:
                     # Basic check for unhashable types before converting
                     is_unhashable = False
                     try:
                         # More robust check if needed, but can be slow:
                         # is_unhashable = df[col].apply(lambda x: isinstance(x, (list, dict))).any()
                         # Using a sample check for performance:
                         sample_check = df[col].iloc[:min(1000, len(df[col]))].apply(lambda x: isinstance(x, (list, dict))).any()
                         if not sample_check: # If sample is okay, attempt conversion
                            df[col] = df[col].astype('category')
                     except Exception: pass # Ignore errors during check/conversion
            except TypeError: pass # Contains unhashable types
            except Exception: pass # Other errors during unique()
    return df

# --- Modified process_json_group_data (with json_normalize reinstated) ---
def process_json_group_data(json_data, group_lookup):
    """
    Processes group data, updates lookup, flattens feed items using
    json_normalize, and returns LISTS of feed item dictionaries.
    """
    all_feed_items_list = [] # Accumulates feed items for the *entire file*
    if group_lookup is None: group_lookup = {}

    for group_id, group_data in json_data.items():
        current_members = [item['id'] for item in group_data.get('members', []) if isinstance(item, dict) and 'id' in item]
        admin_members = [item['id'] for item in group_data.get('admins', []) if isinstance(item, dict) and 'id' in item]
        former_members = [item['id'] for item in group_data.get('former_members', []) if isinstance(item, dict) and 'id' in item]
        feed_ids = []

        # Process feed items for THIS group
        feed_items_raw = group_data.get('feed', [])
        if isinstance(feed_items_raw, list):
            valid_feed_items = []
            for item in feed_items_raw:
                if not (isinstance(item, dict) and 'id' in item): continue

                feed_ids.append(item['id'])
                processed_item = item.copy() # Work on a copy for pre-processing

                # Pre-process 'properties' before normalization if needed (as per original logic)
                properties_raw = processed_item.get('properties')
                if isinstance(properties_raw, list) and len(properties_raw) > 0:
                    props = properties_raw[0]
                    if isinstance(props, dict):
                        for key, value in props.items():
                            # Add pre-processed properties directly. json_normalize might handle others.
                            processed_item[f'properties_{key}'] = value
                    if 'properties' in processed_item: del processed_item['properties'] # Avoid redundant processing

                valid_feed_items.append(processed_item)

            # *** Apply json_normalize to the list for THIS group ***
            if valid_feed_items:
                try:
                    # Normalize the list of feed items for the current group
                    group_feed_df_chunk = pd.json_normalize(valid_feed_items, sep='_')

                    # Add group_id column
                    group_feed_df_chunk['group_id'] = group_id

                    # Convert the resulting DataFrame chunk back to a list of dictionaries
                    chunk_dicts = group_feed_df_chunk.to_dict(orient='records')

                    # Extend the list for the entire file
                    all_feed_items_list.extend(chunk_dicts)

                except Exception as e:
                    print(f"  Warning: json_normalize failed for feed in group {group_id}: {e}")
                    # Optionally append raw items if normalization fails:
                    # for item in valid_feed_items: item['group_id'] = group_id
                    # all_feed_items_list.extend(valid_feed_items)


        # Update group lookup (same as before)
        if group_id not in group_lookup: group_lookup[group_id] = {}
        group_lookup[group_id]['current_members'] = current_members or None
        group_lookup[group_id]['admin_members'] = admin_members or None
        group_lookup[group_id]['former_members'] = former_members or None
        group_lookup[group_id]['feed_ids'] = feed_ids or None

    # Return the list accumulated from all groups in this file
    return group_lookup, all_feed_items_list

# --- Modified process_json_member_conversation_data (with json_normalize reinstated) ---
def process_json_member_conversation_data(json_data, member_lookup):
    """
    Processes member conversation messages, updates lookup, flattens messages
    using json_normalize, and returns LIST of message item dictionaries.
    """
    all_messages_list = [] # Accumulates messages for the *entire file*
    if member_lookup is None: member_lookup = {}

    for member_id, member_conversations in json_data.items():
        if member_id not in member_lookup: member_lookup[member_id] = {}
        existing_conv_ids = member_lookup[member_id].get('conversation_ids', [])
        if existing_conv_ids is None: existing_conv_ids = []
        if not isinstance(existing_conv_ids, list): existing_conv_ids = [existing_conv_ids]
        existing_conv_ids_set = set(existing_conv_ids)

        # Process each conversation for THIS member
        for conversation_id, messages_raw in member_conversations.items():
            existing_conv_ids_set.add(conversation_id)

            if not isinstance(messages_raw, list) or len(messages_raw) == 0: continue

            # Pre-process messages (e.g., tags) before normalization
            valid_messages = []
            for message in messages_raw:
                if not isinstance(message, dict): continue
                processed_message = message.copy()

                # Handle tags (as per original logic)
                tags_data = processed_message.get('tags', {}).get('data')
                if isinstance(tags_data, list):
                    tag_names = [tag.get('name') for tag in tags_data if isinstance(tag, dict) and 'name' in tag]
                    processed_message['tag_names'] = tag_names if tag_names else None
                if 'tags' in processed_message: del processed_message['tags'] # Avoid redundant processing

                valid_messages.append(processed_message)

            # *** Apply json_normalize to the list of messages for THIS conversation ***
            if valid_messages:
                try:
                    conv_messages_df_chunk = pd.json_normalize(valid_messages, sep='_')

                    # Add member and conversation IDs
                    conv_messages_df_chunk['member_id'] = member_id
                    conv_messages_df_chunk['conversation_id'] = conversation_id

                    # Handle 'from_community_id' rename manually if needed (normalize should mostly handle it)
                    # Example check (adjust if normalize doesn't produce desired column name):
                    # if 'from.community.id' in conv_messages_df_chunk.columns:
                    #    conv_messages_df_chunk.rename(columns={'from.community.id': 'from_community_id'}, inplace=True)

                    # Convert DataFrame chunk back to list of dictionaries
                    chunk_dicts = conv_messages_df_chunk.to_dict(orient='records')

                    # Extend the list for the entire file
                    all_messages_list.extend(chunk_dicts)

                except Exception as e:
                    print(f"  Warning: json_normalize failed for messages in convo {conversation_id}, member {member_id}: {e}")
                    # Optionally append raw items:
                    # for item in valid_messages:
                    #     item['member_id'] = member_id; item['conversation_id'] = conversation_id
                    # all_messages_list.extend(valid_messages)

        # Update conversation_ids in lookup (same as before)
        member_lookup[member_id]['conversation_ids'] = list(existing_conv_ids_set) or None

    # Return the list accumulated from all members/convos in this file
    return member_lookup, all_messages_list

# --- Modified process_json_member_data (with json_normalize reinstated for feeds) ---
def process_json_member_data(json_data, member_lookup):
    """
    Processes member data, updates lookup, flattens member feeds using
    json_normalize, manually creates conversation metadata, and returns LISTS
    of member feed dictionaries and conversation metadata dictionaries.
    """
    all_member_feeds_list = [] # Accumulates feeds for the *entire file*
    all_conversations_metadata_list = [] # Accumulates convo metadata for *entire file*
    if member_lookup is None: member_lookup = {}

    for member_id, member_data in json_data.items():
        if member_id not in member_lookup: member_lookup[member_id] = {}
        feed_ids_chunk, manager_ids_chunk, group_ids_chunk, conversation_ids_chunk = [], [], [], []
        existing_conv_ids = member_lookup[member_id].get('conversation_ids', [])
        if existing_conv_ids is None: existing_conv_ids = []
        if not isinstance(existing_conv_ids, list): existing_conv_ids = [existing_conv_ids]
        existing_conv_ids_set = set(existing_conv_ids)

        # --- Process feed items for THIS member ---
        feed_items_raw = member_data.get('feed', [])
        if isinstance(feed_items_raw, list):
            valid_feed_items = []
            for item in feed_items_raw:
                if not (isinstance(item, dict) and 'id' in item): continue
                feed_ids_chunk.append(item['id'])
                processed_item = item.copy()

                # Pre-process 'properties' (as per original logic)
                properties_raw = processed_item.get('properties')
                if isinstance(properties_raw, list) and len(properties_raw) > 0:
                    props = properties_raw[0]
                    if isinstance(props, dict):
                        for key, value in props.items():
                            processed_item[f'properties_{key}'] = value
                    if 'properties' in processed_item: del processed_item['properties']

                valid_feed_items.append(processed_item)

            # *** Apply json_normalize to the list of feed items for THIS member ***
            if valid_feed_items:
                try:
                    member_feed_df_chunk = pd.json_normalize(valid_feed_items, sep='_')

                    # Add member_id
                    member_feed_df_chunk['member_id'] = member_id

                    # Convert DataFrame chunk back to list of dictionaries
                    chunk_dicts = member_feed_df_chunk.to_dict(orient='records')

                    # Extend the list for the entire file
                    all_member_feeds_list.extend(chunk_dicts)

                except Exception as e:
                    print(f"  Warning: json_normalize failed for feed for member {member_id}: {e}")
                    # Optionally append raw items:
                    # for item in valid_feed_items: item['member_id'] = member_id
                    # all_member_feeds_list.extend(valid_feed_items)

        # --- Process conversations metadata (Manual creation - kept as is) ---
        conversations_raw = member_data.get('conversations', [])
        if isinstance(conversations_raw, list):
            for conv in conversations_raw:
                if not (isinstance(conv, dict) and 'id' in conv): continue
                conv_id = conv['id']
                conversation_ids_chunk.append(conv_id)
                existing_conv_ids_set.add(conv_id)

                participant_ids = []
                participants_data = conv.get('participants', {}).get('data')
                if isinstance(participants_data, list):
                    participant_ids = [p.get('id') for p in participants_data if isinstance(p, dict) and 'id' in p]

                conversation_record = {
                    'id': conv_id, 'member_id': member_id,
                    'participant_ids': participant_ids or None,
                    'message_count': conv.get('message_count'),
                    'unread_count': conv.get('unread_count'),
                    'updated_time': conv.get('updated_time')
                }
                # Add this manually created record to the list for the file
                all_conversations_metadata_list.append(conversation_record)

        # --- Process managers and groups lists (same as before) ---
        managers_raw = member_data.get('managers', [])
        if isinstance(managers_raw, list):
            manager_ids_chunk = [m['id'] for m in managers_raw if isinstance(m, dict) and 'id' in m]
        groups_raw = member_data.get('groups', [])
        if isinstance(groups_raw, list):
            group_ids_chunk = [g['id'] for g in groups_raw if isinstance(g, dict) and 'id' in g]

        # --- Update member_lookup (same as before) ---
        member_lookup[member_id]['feed_ids'] = feed_ids_chunk or None
        member_lookup[member_id]['manager_ids'] = manager_ids_chunk or None
        member_lookup[member_id]['group_ids'] = group_ids_chunk or None
        member_lookup[member_id]['conversation_ids'] = list(existing_conv_ids_set) or None

    # Return the lists accumulated from all members in this file
    return member_lookup, all_member_feeds_list, all_conversations_metadata_list


# --- Main Extraction Function (largely the same, uses updated process functions) ---
def extract_data(data_dir):
    output_dir = data_dir + '_converted_parquet'
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    print(f"Output will be saved to {output_dir}")

    groups_final_path = os.path.join(output_dir, 'groups.parquet')
    members_final_path = os.path.join(output_dir, 'members.parquet')
    group_feeds_base_path = os.path.join(output_dir, 'group_feeds')
    member_feeds_base_path = os.path.join(output_dir, 'member_feeds')
    conversations_base_path = os.path.join(output_dir, 'conversations') # Convo metadata
    messages_base_path = os.path.join(output_dir, 'member_messages') # Actual messages

    # --- Load initial static groups and members data (same as before) ---
    print("Reading initial groups and members data...")
    groups_only_data_path = os.path.join(data_dir, 'groups.json')
    members_only_data_path = os.path.join(data_dir, 'members.json')
    groups_static_df = pd.DataFrame()
    if os.path.exists(groups_only_data_path):
        try:
            with open(groups_only_data_path, 'r', encoding='utf-8') as f: groups_only_data = json.load(f)
            groups_static_df = pd.json_normalize(groups_only_data) # Normalize static data too
        except Exception as e: print(f"Warning: Could not read/parse {groups_only_data_path}: {e}")
    else: print(f"Warning: {groups_only_data_path} not found.")

    members_static_df = pd.DataFrame()
    if os.path.exists(members_only_data_path):
        try:
            with open(members_only_data_path, 'r', encoding='utf-8') as f: members_only_data = json.load(f)
            members_dfs = []
            admin_ids = set()
            # Use normalize for static member lists too, for consistency
            if "admins" in members_only_data and isinstance(members_only_data["admins"], list):
                admin_members_df = pd.json_normalize(members_only_data["admins"])
                if not admin_members_df.empty:
                    admin_members_df["member_type"] = "admin"
                    members_dfs.append(admin_members_df)
                    admin_ids = set(admin_members_df['id'].tolist())

            processed_ids = admin_ids.copy()
            if "members" in members_only_data and isinstance(members_only_data["members"], list):
                current_members_df = pd.json_normalize(members_only_data["members"])
                if not current_members_df.empty:
                    current_members_df = current_members_df[~current_members_df['id'].isin(admin_ids)] # Exclude admins
                    current_members_df["member_type"] = "current"
                    members_dfs.append(current_members_df)
                    processed_ids.update(set(current_members_df['id'].tolist()))

            if "former_members" in members_only_data and isinstance(members_only_data["former_members"], list):
                former_members_df = pd.json_normalize(members_only_data["former_members"])
                if not former_members_df.empty:
                    former_members_df = former_members_df[~former_members_df['id'].isin(processed_ids)] # Exclude admins/current
                    former_members_df["member_type"] = "former"
                    members_dfs.append(former_members_df)

            if members_dfs: members_static_df = pd.concat(members_dfs, ignore_index=True)
            del members_dfs, members_only_data
        except Exception as e: print(f"Warning: Could not read/parse {members_only_data_path}: {e}")
    else: print(f"Warning: {members_only_data_path} not found.")

    # --- Initialize Lookups (same as before) ---
    group_lookup = {}
    member_lookup = {}
    if not groups_static_df.empty and 'id' in groups_static_df.columns:
        for gid in groups_static_df['id']: group_lookup[gid] = {}
    if not members_static_df.empty and 'id' in members_static_df.columns:
         for mid in members_static_df['id']: member_lookup[mid] = {}

    # --- Find and Process JSON files (same as before) ---
    all_json_files = []
    for root, dirs, files in os.walk(data_dir):
        dirs[:] = [d for d in dirs if os.path.join(root, d) != output_dir]
        for file in files:
            if file.endswith('.json') and file not in ['groups.json', 'members.json']:
                all_json_files.append(os.path.join(root, file))

    print(f"Found {len(all_json_files)} JSON data files to process.")
    pbar = tqdm(all_json_files, desc="Processing files", unit="file")
    chunk_counter = 0

    for file_path in pbar:
        parent_dir = os.path.basename(os.path.dirname(file_path))
        file_name = os.path.basename(file_path)
        parent_dir_type = 'unknown'
        if parent_dir == 'group-data': parent_dir_type = 'group-data'
        elif parent_dir == 'member-data': parent_dir_type = 'member-data'
        elif parent_dir == 'conversation-messages':
            grand_parent_dir = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
            if grand_parent_dir == 'member-data': parent_dir_type = 'conversation-messages'

        pbar.set_description(f"Processing {parent_dir}/{file_name}")

        # Initialize lists to hold data extracted *from this file*
        file_group_feeds_list, file_member_feeds_list = [], []
        file_conversations_list, file_messages_list = [], []

        try:
            with open(file_path, 'r', encoding='utf-8') as f: json_data = json.load(f)

            # --- Call appropriate processing function ---
            # These functions now return the lookups and the LISTS of dicts
            if parent_dir_type == 'group-data':
                group_lookup, file_group_feeds_list = process_json_group_data(json_data, group_lookup)
            elif parent_dir_type == 'member-data':
                member_lookup, file_member_feeds_list, file_conversations_list = process_json_member_data(json_data, member_lookup)
            elif parent_dir_type == 'conversation-messages':
                member_lookup, file_messages_list = process_json_member_conversation_data(json_data, member_lookup)
            else: pass # Skip unknown

            # --- Write extracted lists from THIS FILE to Parquet chunks ---
            chunk_counter += 1
            chunk_suffix = f"_{chunk_counter:06d}.parquet"

            # Write Group Feeds Chunk (if any produced by this file)
            if file_group_feeds_list:
                df_chunk = pd.DataFrame(file_group_feeds_list)
                df_chunk = optimize_dtypes(df_chunk)
                table_chunk = pa.Table.from_pandas(df_chunk, schema=None, preserve_index=False)
                pq.write_table(table_chunk, group_feeds_base_path + chunk_suffix)
                del df_chunk, table_chunk

            # Write Member Feeds Chunk
            if file_member_feeds_list:
                df_chunk = pd.DataFrame(file_member_feeds_list)
                df_chunk = optimize_dtypes(df_chunk)
                table_chunk = pa.Table.from_pandas(df_chunk, schema=None, preserve_index=False)
                pq.write_table(table_chunk, member_feeds_base_path + chunk_suffix)
                del df_chunk, table_chunk

            # Write Conversations Metadata Chunk
            if file_conversations_list:
                df_chunk = pd.DataFrame(file_conversations_list)
                df_chunk = optimize_dtypes(df_chunk)
                table_chunk = pa.Table.from_pandas(df_chunk, schema=None, preserve_index=False)
                pq.write_table(table_chunk, conversations_base_path + chunk_suffix)
                del df_chunk, table_chunk

            # Write Messages Chunk
            if file_messages_list:
                df_chunk = pd.DataFrame(file_messages_list)
                df_chunk = optimize_dtypes(df_chunk)
                table_chunk = pa.Table.from_pandas(df_chunk, schema=None, preserve_index=False)
                pq.write_table(table_chunk, messages_base_path + chunk_suffix)
                del df_chunk, table_chunk

            # Clean up memory
            del json_data
            # Ensure lists are deleted even if empty
            del file_group_feeds_list, file_member_feeds_list
            del file_conversations_list, file_messages_list
            gc.collect()

        except json.JSONDecodeError as e: print(f"\nError decoding JSON in file {file_path}: {e}")
        except Exception as e: print(f"\nError processing file {file_path}: {e}")

    pbar.close()
    if chunk_counter == 0: print("Warning: No data files processed or no data extracted.")
    print("File processing complete!")

    # --- Finalize and Save groups_df and members_df (same as before) ---
    print("Finalizing and saving groups and members data...")
    groups_dynamic_df = pd.DataFrame.from_dict(group_lookup, orient='index')
    groups_dynamic_df.index.name = 'id'; groups_dynamic_df.reset_index(inplace=True)
    members_dynamic_df = pd.DataFrame.from_dict(member_lookup, orient='index')
    members_dynamic_df.index.name = 'id'; members_dynamic_df.reset_index(inplace=True)

    groups_final_df = pd.DataFrame()
    if not groups_static_df.empty:
        if 'id' in groups_static_df.columns and 'id' in groups_dynamic_df.columns:
             try: groups_static_df['id'] = groups_static_df['id'].astype(groups_dynamic_df['id'].dtype)
             except Exception: pass
             # Use outer merge to keep all static groups and add dynamic info where available
             groups_final_df = pd.merge(groups_static_df, groups_dynamic_df, on='id', how='outer')
        else: groups_final_df = groups_dynamic_df
    elif not groups_dynamic_df.empty: groups_final_df = groups_dynamic_df

    members_final_df = pd.DataFrame()
    if not members_static_df.empty:
        if 'id' in members_static_df.columns and 'id' in members_dynamic_df.columns:
             try: members_static_df['id'] = members_static_df['id'].astype(members_dynamic_df['id'].dtype)
             except Exception: pass
             # Use outer merge to keep all static members
             members_final_df = pd.merge(members_static_df, members_dynamic_df, on='id', how='outer')
        else: members_final_df = members_dynamic_df
    elif not members_dynamic_df.empty: members_final_df = members_dynamic_df

    if not groups_final_df.empty:
        groups_final_df = optimize_dtypes(groups_final_df)
        groups_final_df.to_parquet(groups_final_path, engine='pyarrow', index=False)
        print(f"Saved final groups data to {groups_final_path}")
    else: print("No final groups data to save.")

    if not members_final_df.empty:
        members_final_df = optimize_dtypes(members_final_df)
        members_final_df.to_parquet(members_final_path, engine='pyarrow', index=False)
        print(f"Saved final members data to {members_final_path}")
    else: print("No final members data to save.")

    del groups_final_df, members_final_df, groups_dynamic_df, members_dynamic_df, groups_static_df, members_static_df
    del group_lookup, member_lookup
    gc.collect()

    print(f"\nData extraction process finished!")
    print(f"Output saved in: {output_dir}")
    print(f"You can read combined data using pd.read_parquet('{os.path.join(output_dir, 'group_feeds_*.parquet')}') or Dask.")


# --- Main Execution Block ---
if __name__ == "__main__":
    # >>>>> IMPORTANT: SET YOUR DATA DIRECTORY HERE <<<<<
    data_dir = "Bunnings" # <--- EXAMPLE: Set your input directory

    if not os.path.isdir(data_dir):
        print(f"Error: Directory not found: {data_dir}")
        print("Please ensure the 'data_dir' variable points to the correct location.")
    else:
        extract_data(data_dir)