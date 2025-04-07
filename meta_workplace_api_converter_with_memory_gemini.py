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
def optimize_dtypes(df):
    if df.empty: return df
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == 'object':
            try:
                # Basic check for unhashable types before attempting category conversion
                is_unhashable_sample = False
                sample_size = min(1000, len(df[col]))
                if sample_size > 0:
                    # Check if any item in the sample is a list or dict
                    is_unhashable_sample = df[col].iloc[:sample_size].apply(lambda x: isinstance(x, (list, dict))).any()

                if not is_unhashable_sample:
                    num_unique_values = len(df[col].unique())
                    num_total_values = len(df[col])
                    # Convert to category if low cardinality or few unique values, and seems hashable
                    if num_total_values > 0 and (num_unique_values / num_total_values < 0.1):
                       df[col] = df[col].astype('category')
            except TypeError: pass # Contains unhashable types, skip category conversion
            except Exception: pass # Other errors during unique() or type checking, skip
        # Example: Convert potential datetime strings (optional, can be done later)
        elif 'time' in col.lower() or 'date' in col.lower():
            try:
                # Attempt conversion, coerce errors to NaT (Not a Time)
                df[col] = pd.to_datetime(df[col], errors='coerce')
                # Optional: Convert to Arrow timestamp if needed for Parquet efficiency
                # if pd.api.types.is_datetime64_any_dtype(df[col]):
                #    df[col] = pd.Series(pa.array(df[col].astype(int) // 10**6, type=pa.timestamp('ms'))) # Example ms
            except Exception: pass # Ignore conversion errors
    return df

# --- Processing Function for Group Data (Feed items) ---
# (No changes needed in this function)
def process_json_group_data(json_data, group_lookup):
    all_feed_items_list = [] # Accumulates feed items for the *entire file*
    if group_lookup is None: group_lookup = {}

    for group_id, group_data in json_data.items():
        if not isinstance(group_data, dict): continue # Skip if group_data isn't a dict

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
                processed_item = item.copy()
                properties_raw = processed_item.get('properties')
                if isinstance(properties_raw, list) and len(properties_raw) > 0:
                    props = properties_raw[0]
                    if isinstance(props, dict):
                        for key, value in props.items():
                            processed_item[f'properties_{key}'] = value
                    if 'properties' in processed_item: del processed_item['properties']
                valid_feed_items.append(processed_item)

            if valid_feed_items:
                try:
                    group_feed_df_chunk = pd.json_normalize(valid_feed_items, sep='_')
                    group_feed_df_chunk['group_id'] = group_id
                    chunk_dicts = group_feed_df_chunk.to_dict(orient='records')
                    all_feed_items_list.extend(chunk_dicts)
                except Exception as e:
                    print(f"  Warning: json_normalize failed for feed in group {group_id}: {e}")

        if group_id not in group_lookup: group_lookup[group_id] = {}
        group_lookup[group_id]['current_members'] = current_members or None
        group_lookup[group_id]['admin_members'] = admin_members or None
        group_lookup[group_id]['former_members'] = former_members or None
        group_lookup[group_id]['feed_ids'] = feed_ids or None

    return group_lookup, all_feed_items_list

# --- Processing Function for Member Conversation Messages ---
# (No changes needed in this function)
def process_json_member_conversation_data(json_data, member_lookup):
    all_messages_list = [] # Accumulates messages for the *entire file*
    if member_lookup is None: member_lookup = {}

    for member_id, member_conversations in json_data.items():
        if not isinstance(member_conversations, dict): continue # Skip if not a dict of conversations

        if member_id not in member_lookup: member_lookup[member_id] = {}
        existing_conv_ids = member_lookup[member_id].get('conversation_ids', [])
        if existing_conv_ids is None: existing_conv_ids = []
        if not isinstance(existing_conv_ids, list): existing_conv_ids = [existing_conv_ids] # Ensure it's a list
        existing_conv_ids_set = set(existing_conv_ids)

        for conversation_id, messages_raw in member_conversations.items():
            existing_conv_ids_set.add(conversation_id)
            if not isinstance(messages_raw, list) or len(messages_raw) == 0: continue

            valid_messages = []
            for message in messages_raw:
                if not isinstance(message, dict): continue
                processed_message = message.copy()
                tags_data = processed_message.get('tags', {}).get('data')
                if isinstance(tags_data, list):
                    tag_names = [tag.get('name') for tag in tags_data if isinstance(tag, dict) and 'name' in tag]
                    processed_message['tag_names'] = tag_names if tag_names else None
                if 'tags' in processed_message: del processed_message['tags']
                valid_messages.append(processed_message)

            if valid_messages:
                try:
                    conv_messages_df_chunk = pd.json_normalize(valid_messages, sep='_')
                    conv_messages_df_chunk['member_id'] = member_id
                    conv_messages_df_chunk['conversation_id'] = conversation_id
                    chunk_dicts = conv_messages_df_chunk.to_dict(orient='records')
                    all_messages_list.extend(chunk_dicts)
                except Exception as e:
                    print(f"  Warning: json_normalize failed for messages in convo {conversation_id}, member {member_id}: {e}")

        member_lookup[member_id]['conversation_ids'] = list(existing_conv_ids_set) or None

    return member_lookup, all_messages_list

# --- Processing Function for Member Data (Feed, Conversation Metadata, AND Events) ---
def process_json_member_data(json_data, member_lookup):
    """
    Processes member data: updates lookup, flattens feeds and events using
    json_normalize, manually creates conversation metadata. Returns LISTS
    of member feed dicts, conversation metadata dicts, and member event dicts.
    """
    all_member_feeds_list = []
    all_conversations_metadata_list = []
    all_member_events_list = [] # <<< NEW list for events
    if member_lookup is None: member_lookup = {}

    for member_id, member_data in json_data.items():
        if not isinstance(member_data, dict): continue # Skip if member_data isn't a dict

        if member_id not in member_lookup: member_lookup[member_id] = {}
        feed_ids_chunk, manager_ids_chunk, group_ids_chunk, conversation_ids_chunk = [], [], [], []
        existing_conv_ids = member_lookup[member_id].get('conversation_ids', [])
        if existing_conv_ids is None: existing_conv_ids = []
        if not isinstance(existing_conv_ids, list): existing_conv_ids = [existing_conv_ids]
        existing_conv_ids_set = set(existing_conv_ids)

        # --- Process feed items for THIS member (Same as before) ---
        feed_items_raw = member_data.get('feed', [])
        if isinstance(feed_items_raw, list):
            valid_feed_items = []
            for item in feed_items_raw:
                if not (isinstance(item, dict) and 'id' in item): continue
                feed_ids_chunk.append(item['id'])
                processed_item = item.copy()
                properties_raw = processed_item.get('properties')
                if isinstance(properties_raw, list) and len(properties_raw) > 0:
                    props = properties_raw[0]
                    if isinstance(props, dict):
                        for key, value in props.items():
                            processed_item[f'properties_{key}'] = value
                    if 'properties' in processed_item: del processed_item['properties']
                valid_feed_items.append(processed_item)

            if valid_feed_items:
                try:
                    member_feed_df_chunk = pd.json_normalize(valid_feed_items, sep='_')
                    member_feed_df_chunk['member_id'] = member_id
                    chunk_dicts = member_feed_df_chunk.to_dict(orient='records')
                    all_member_feeds_list.extend(chunk_dicts)
                except Exception as e:
                    print(f"  Warning: json_normalize failed for feed for member {member_id}: {e}")

        # --- Process conversations metadata (Manual creation - Same as before) ---
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
                all_conversations_metadata_list.append(conversation_record)

        # --- NEW: Process events for THIS member ---
        events_raw = member_data.get('events', [])
        if isinstance(events_raw, list) and events_raw:
            # Filter for valid dicts, although normalize might handle some errors
            valid_events = [evt for evt in events_raw if isinstance(evt, dict)]
            if valid_events:
                try:
                    member_events_df_chunk = pd.json_normalize(valid_events, sep='_')
                    member_events_df_chunk['member_id'] = member_id # Add member ID

                    # Handle potential nested structures remaining after normalize (convert to string)
                    for col in member_events_df_chunk.columns:
                        # Check a sample for performance
                        sample_size = min(100, len(member_events_df_chunk[col]))
                        if sample_size > 0 and member_events_df_chunk[col].iloc[:sample_size].apply(lambda x: isinstance(x, (list, dict))).any():
                            try:
                                member_events_df_chunk[col] = member_events_df_chunk[col].astype(str) # Convert complex types to string
                            except Exception:
                                pass # Ignore conversion errors for specific columns

                    chunk_dicts = member_events_df_chunk.to_dict(orient='records')
                    all_member_events_list.extend(chunk_dicts)
                except Exception as e:
                    print(f"  Warning: json_normalize failed for events for member {member_id}: {e}")


        # --- Process managers and groups lists (Same as before) ---
        managers_raw = member_data.get('managers', [])
        if isinstance(managers_raw, list):
            manager_ids_chunk = [m['id'] for m in managers_raw if isinstance(m, dict) and 'id' in m]
        groups_raw = member_data.get('groups', [])
        if isinstance(groups_raw, list):
            group_ids_chunk = [g['id'] for g in groups_raw if isinstance(g, dict) and 'id' in g]

        # --- Update member_lookup (Same as before) ---
        member_lookup[member_id]['feed_ids'] = feed_ids_chunk or None
        member_lookup[member_id]['manager_ids'] = manager_ids_chunk or None
        member_lookup[member_id]['group_ids'] = group_ids_chunk or None
        member_lookup[member_id]['conversation_ids'] = list(existing_conv_ids_set) or None
        # Note: We don't store event_ids in the lookup by default, but could if needed.

    # <<< Return the new events list as well
    return member_lookup, all_member_feeds_list, all_conversations_metadata_list, all_member_events_list


# --- Processing Function for Post Data (Comments, Reactions, Seen) ---
# (No changes needed in this function)
def process_json_post_data(json_data):
    """
    Processes post data (comments, reactions, seen), flattens comments
    using json_normalize, and returns separate LISTS of dictionaries for each.
    """
    all_comments_list = []
    all_reactions_list = []
    all_seen_list = []

    for post_id, post_content in json_data.items():
        if not isinstance(post_content, dict): continue # Skip if post_content isn't a dict

        # --- Process Comments ---
        comments_raw = post_content.get('comments', [])
        if isinstance(comments_raw, list) and comments_raw:
            valid_comments = [c for c in comments_raw if isinstance(c, dict)] # Basic validation
            if valid_comments:
                try:
                    comments_df_chunk = pd.json_normalize(valid_comments, sep='_')
                    comments_df_chunk['post_id'] = post_id
                    # Handle potential list types after normalize (e.g., message_tags)
                    for col in comments_df_chunk.columns:
                        # More robust check across the whole column if performance allows, else sample
                        if comments_df_chunk[col].dtype == 'object' and comments_df_chunk[col].apply(lambda x: isinstance(x, list)).any():
                             comments_df_chunk[col] = comments_df_chunk[col].astype(str) # Convert list cols to string for Parquet
                    chunk_dicts = comments_df_chunk.to_dict(orient='records')
                    all_comments_list.extend(chunk_dicts)
                except Exception as e:
                    print(f"  Warning: json_normalize failed for comments in post {post_id}: {e}")

        # --- Process Reactions ---
        reactions_raw = post_content.get('reactions', [])
        if isinstance(reactions_raw, list) and reactions_raw:
            valid_reactions = [r for r in reactions_raw if isinstance(r, dict)]
            if valid_reactions:
                try:
                    reactions_df_chunk = pd.DataFrame(valid_reactions)
                    reactions_df_chunk['post_id'] = post_id
                    chunk_dicts = reactions_df_chunk.to_dict(orient='records')
                    all_reactions_list.extend(chunk_dicts)
                except Exception as e:
                     print(f"  Warning: Failed processing reactions for post {post_id}: {e}")

        # --- Process Seen ---
        seen_raw = post_content.get('seen', [])
        if isinstance(seen_raw, list) and seen_raw:
            valid_seen = [s for s in seen_raw if isinstance(s, dict)]
            if valid_seen:
                try:
                    seen_df_chunk = pd.DataFrame(valid_seen)
                    seen_df_chunk['post_id'] = post_id
                    chunk_dicts = seen_df_chunk.to_dict(orient='records')
                    all_seen_list.extend(chunk_dicts)
                except Exception as e:
                    print(f"  Warning: Failed processing seen for post {post_id}: {e}")

    return all_comments_list, all_reactions_list, all_seen_list

# --- Processing Function for Post Summaries ---
# (No changes needed in this function)
def process_json_post_summaries(json_data):
    """
    Processes post summary data, flattens using json_normalize,
    and returns a LIST of summary dictionaries.
    """
    all_summaries_list = []

    for post_id, post_content in json_data.items():
        if isinstance(post_content, dict):
             normalize_input = [post_content]
        elif isinstance(post_content, list):
             normalize_input = post_content # Allow list of summaries per post ID if structure varies
        else:
             continue

        if normalize_input:
            try:
                summary_df_chunk = pd.json_normalize(normalize_input, sep='_')
                summary_df_chunk['post_id'] = post_id
                # Handle potential list/dict types after normalize if necessary
                for col in summary_df_chunk.columns:
                    sample_size = min(100, len(summary_df_chunk[col]))
                    if sample_size > 0 and summary_df_chunk[col].iloc[:sample_size].apply(lambda x: isinstance(x, (list, dict))).any():
                        try:
                            summary_df_chunk[col] = summary_df_chunk[col].astype(str) # Convert complex types to string
                        except Exception:
                            pass
                chunk_dicts = summary_df_chunk.to_dict(orient='records')
                all_summaries_list.extend(chunk_dicts)
            except Exception as e:
                print(f"  Warning: json_normalize failed for summary for post {post_id}: {e}")

    return all_summaries_list


# --- Main Extraction Function (Updated) ---
def extract_data(data_dir):
    output_dir = data_dir + '_converted_parquet'
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    print(f"Output will be saved to {output_dir}")

    # Define output paths
    groups_final_path = os.path.join(output_dir, 'groups.parquet')
    members_final_path = os.path.join(output_dir, 'members.parquet')
    group_feeds_base_path = os.path.join(output_dir, 'group_feeds')
    member_feeds_base_path = os.path.join(output_dir, 'member_feeds')
    conversations_base_path = os.path.join(output_dir, 'conversations') # Convo metadata
    messages_base_path = os.path.join(output_dir, 'member_messages') # Actual messages
    post_comments_base_path = os.path.join(output_dir, 'post_comments')
    post_reactions_base_path = os.path.join(output_dir, 'post_reactions')
    post_seen_base_path = os.path.join(output_dir, 'post_seen')
    post_summaries_base_path = os.path.join(output_dir, 'post_summaries')
    member_events_base_path = os.path.join(output_dir, 'member_events') # <<< NEW output path for events


    # --- Load initial static groups and members data (Same as before) ---
    print("Reading initial groups and members data...")
    groups_only_data_path = os.path.join(data_dir, 'groups.json')
    members_only_data_path = os.path.join(data_dir, 'members.json')
    groups_static_df = pd.DataFrame()
    if os.path.exists(groups_only_data_path):
        try:
            with open(groups_only_data_path, 'r', encoding='utf-8') as f: groups_only_data = json.load(f)
            if isinstance(groups_only_data, list):
                groups_static_df = pd.json_normalize(groups_only_data)
            elif isinstance(groups_only_data, dict):
                 groups_static_df = pd.DataFrame.from_dict(groups_only_data, orient='index').reset_index().rename(columns={'index': 'id'})
        except Exception as e: print(f"Warning: Could not read/parse {groups_only_data_path}: {e}")
    else: print(f"Warning: {groups_only_data_path} not found.")

    members_static_df = pd.DataFrame()
    if os.path.exists(members_only_data_path):
        try:
            with open(members_only_data_path, 'r', encoding='utf-8') as f: members_only_data = json.load(f)
            members_dfs = []
            admin_ids = set()
            if "admins" in members_only_data and isinstance(members_only_data["admins"], list):
                admin_members_df = pd.json_normalize(members_only_data["admins"])
                if not admin_members_df.empty:
                    admin_members_df["member_type"] = "admin"
                    members_dfs.append(admin_members_df)
                    if 'id' in admin_members_df.columns: admin_ids = set(admin_members_df['id'].tolist())

            processed_ids = admin_ids.copy()
            if "members" in members_only_data and isinstance(members_only_data["members"], list):
                current_members_df = pd.json_normalize(members_only_data["members"])
                if not current_members_df.empty and 'id' in current_members_df.columns:
                    current_members_df = current_members_df[~current_members_df['id'].isin(admin_ids)]
                    current_members_df["member_type"] = "current"
                    members_dfs.append(current_members_df)
                    processed_ids.update(set(current_members_df['id'].tolist()))

            if "former_members" in members_only_data and isinstance(members_only_data["former_members"], list):
                former_members_df = pd.json_normalize(members_only_data["former_members"])
                if not former_members_df.empty and 'id' in former_members_df.columns:
                    former_members_df = former_members_df[~former_members_df['id'].isin(processed_ids)]
                    former_members_df["member_type"] = "former"
                    members_dfs.append(former_members_df)

            if members_dfs: members_static_df = pd.concat(members_dfs, ignore_index=True)
            del members_dfs, members_only_data
        except Exception as e: print(f"Warning: Could not read/parse {members_only_data_path}: {e}")
    else: print(f"Warning: {members_only_data_path} not found.")

    # --- Initialize Lookups (Same as before) ---
    group_lookup = {}
    member_lookup = {}
    if not groups_static_df.empty and 'id' in groups_static_df.columns:
        for gid in groups_static_df['id'].dropna().unique(): group_lookup[gid] = {}
    if not members_static_df.empty and 'id' in members_static_df.columns:
         for mid in members_static_df['id'].dropna().unique(): member_lookup[mid] = {}

    # --- Find and Process JSON files (Corrected os.walk logic) ---
    all_json_files = []
    for root, dirs, files in os.walk(data_dir, topdown=True):
        dirs[:] = [d for d in dirs if os.path.join(root, d) != output_dir] # Exclude output dir

        for file in files:
            file_full_path = os.path.join(root, file)
            if file.endswith('.json') and file_full_path not in [groups_only_data_path, members_only_data_path]:
                 all_json_files.append(file_full_path)

    print(f"Found {len(all_json_files)} JSON data files to process.")
    pbar = tqdm(all_json_files, desc="Processing files", unit="file")
    chunk_counter = 0

    # --- Loop through identified JSON files ---
    for file_path in pbar:
        parent_dir = os.path.basename(os.path.dirname(file_path))
        file_name = os.path.basename(file_path)
        try:
            # Calculate grand_parent_dir carefully, handling potential root directory cases
            grand_parent_dir_path = os.path.dirname(os.path.dirname(file_path))
            # Check if grand_parent_dir_path is the same as data_dir before getting basename
            if grand_parent_dir_path != data_dir and os.path.exists(grand_parent_dir_path):
                 grand_parent_dir = os.path.basename(grand_parent_dir_path)
            else:
                 grand_parent_dir = None # Or some indicator that it's directly under data_dir
        except Exception:
             grand_parent_dir = None # Handle potential errors getting parent dirs

        # Determine file type based on directory structure
        parent_dir_type = 'unknown'
        if parent_dir == 'group-data':
            parent_dir_type = 'group-data'
        elif parent_dir == 'member-data':
             # Files directly under member-data (containing feeds, convos, events)
            parent_dir_type = 'member-data'
        elif parent_dir == 'conversation-messages' and grand_parent_dir == 'member-data':
             parent_dir_type = 'conversation-messages'
        elif parent_dir == 'post-data': # Files directly under post-data
             parent_dir_type = 'post-data'
        elif parent_dir == 'post-summaries' and grand_parent_dir == 'post-data': # Files under post-data/post-summaries
             parent_dir_type = 'post-summaries'


        pbar.set_description(f"Processing {os.path.relpath(file_path, data_dir)} (Type: {parent_dir_type})")

        # Initialize lists to hold data extracted *from this file*
        file_group_feeds_list, file_member_feeds_list = [], []
        file_conversations_list, file_messages_list = [], []
        file_comments_list, file_reactions_list, file_seen_list = [], [], []
        file_post_summaries_list = []
        file_member_events_list = [] # <<< Initialize new list for events

        try:
            with open(file_path, 'r', encoding='utf-8') as f: json_data = json.load(f)

            # --- Call appropriate processing function ---
            if parent_dir_type == 'group-data':
                group_lookup, file_group_feeds_list = process_json_group_data(json_data, group_lookup)
            elif parent_dir_type == 'member-data':
                # <<< Update call to receive the new events list
                member_lookup, file_member_feeds_list, file_conversations_list, file_member_events_list = process_json_member_data(json_data, member_lookup)
            elif parent_dir_type == 'conversation-messages':
                member_lookup, file_messages_list = process_json_member_conversation_data(json_data, member_lookup)
            elif parent_dir_type == 'post-data':
                 file_comments_list, file_reactions_list, file_seen_list = process_json_post_data(json_data)
            elif parent_dir_type == 'post-summaries':
                 file_post_summaries_list = process_json_post_summaries(json_data)
            else:
                 pass # Skip unknown types silently

            # --- Write extracted lists from THIS FILE to Parquet chunks ---
            data_written_in_chunk = False
            chunk_suffix = f"_{chunk_counter + 1:06d}.parquet" # Preview next suffix

            def write_chunk(data_list, base_path):
                nonlocal data_written_in_chunk
                if data_list: # Check if the list actually contains data
                    try:
                        df_chunk = pd.DataFrame(data_list)
                        if not df_chunk.empty: # Double check df is not empty
                            df_chunk = optimize_dtypes(df_chunk)
                            table_chunk = pa.Table.from_pandas(df_chunk, schema=None, preserve_index=False)
                            pq.write_table(table_chunk, base_path + chunk_suffix)
                            del df_chunk, table_chunk
                            data_written_in_chunk = True # Mark that we wrote something
                            return True
                    except Exception as write_err:
                        print(f"\n  Error writing chunk to {base_path + chunk_suffix}: {write_err}")
                return False

            # Write chunks for each data type if list is not empty
            write_chunk(file_group_feeds_list, group_feeds_base_path)
            write_chunk(file_member_feeds_list, member_feeds_base_path)
            write_chunk(file_conversations_list, conversations_base_path)
            write_chunk(file_messages_list, messages_base_path)
            write_chunk(file_comments_list, post_comments_base_path)
            write_chunk(file_reactions_list, post_reactions_base_path)
            write_chunk(file_seen_list, post_seen_base_path)
            write_chunk(file_post_summaries_list, post_summaries_base_path)
            write_chunk(file_member_events_list, member_events_base_path) # <<< Write events chunk

            if data_written_in_chunk:
                chunk_counter += 1 # Increment only if data was written

        except json.JSONDecodeError as e: print(f"\nError decoding JSON in file {file_path}: {e}")
        except MemoryError as e: print(f"\nMemoryError processing file {file_path}: {e}. Try reducing chunk size or increasing memory.")
        except Exception as e: print(f"\nError processing file {file_path}: {e}")
        finally:
            # Clean up memory regardless of success or failure inside the loop
            if 'json_data' in locals(): del json_data
            # Ensure all lists created for the file are deleted
            del file_group_feeds_list, file_member_feeds_list
            del file_conversations_list, file_messages_list
            del file_comments_list, file_reactions_list, file_seen_list
            del file_post_summaries_list
            del file_member_events_list # <<< Clean up event list
            gc.collect() # Force garbage collection

    pbar.close()
    if chunk_counter == 0: print("Warning: No data chunks were written. Check input data and directory structure.")
    print(f"File processing complete! {chunk_counter} data chunks written.")

    # --- Finalize and Save groups_df and members_df (Same as before) ---
    print("Finalizing and saving groups and members data...")
    groups_dynamic_df = pd.DataFrame.from_dict(group_lookup, orient='index')
    groups_dynamic_df.index.name = 'id'; groups_dynamic_df.reset_index(inplace=True)
    members_dynamic_df = pd.DataFrame.from_dict(member_lookup, orient='index')
    members_dynamic_df.index.name = 'id'; members_dynamic_df.reset_index(inplace=True)

    # Merge static and dynamic data (robust merge)
    groups_final_df = pd.DataFrame()
    try:
        if not groups_static_df.empty:
            if not groups_dynamic_df.empty and 'id' in groups_static_df.columns and 'id' in groups_dynamic_df.columns:
                 # Attempt to align ID types before merge
                 try:
                     id_col_static_type = groups_static_df['id'].dropna().infer_objects().dtype
                     id_col_dynamic_type = groups_dynamic_df['id'].dropna().infer_objects().dtype
                     if id_col_static_type == id_col_dynamic_type:
                         pass # Types match, proceed
                     else: # Attempt conversion if types differ (e.g., int vs object/string)
                          groups_static_df['id'] = groups_static_df['id'].astype(str)
                          groups_dynamic_df['id'] = groups_dynamic_df['id'].astype(str)
                 except Exception:
                      print("Warning: Could not reliably align group ID types for merging. Converting IDs to string.")
                      groups_static_df['id'] = groups_static_df['id'].astype(str)
                      groups_dynamic_df['id'] = groups_dynamic_df['id'].astype(str)

                 groups_final_df = pd.merge(groups_static_df, groups_dynamic_df, on='id', how='outer')
            else: # Only static exists
                groups_final_df = groups_static_df
        elif not groups_dynamic_df.empty: # Only dynamic exists
            groups_final_df = groups_dynamic_df
    except Exception as e: print(f"Error merging groups data: {e}")

    members_final_df = pd.DataFrame()
    try:
        if not members_static_df.empty:
            if not members_dynamic_df.empty and 'id' in members_static_df.columns and 'id' in members_dynamic_df.columns:
                 # Align ID types
                 try:
                     id_col_static_type = members_static_df['id'].dropna().infer_objects().dtype
                     id_col_dynamic_type = members_dynamic_df['id'].dropna().infer_objects().dtype
                     if id_col_static_type != id_col_dynamic_type:
                          members_static_df['id'] = members_static_df['id'].astype(str)
                          members_dynamic_df['id'] = members_dynamic_df['id'].astype(str)
                 except Exception:
                     print("Warning: Could not reliably align member ID types for merging. Converting IDs to string.")
                     members_static_df['id'] = members_static_df['id'].astype(str)
                     members_dynamic_df['id'] = members_dynamic_df['id'].astype(str)

                 members_final_df = pd.merge(members_static_df, members_dynamic_df, on='id', how='outer')
            else: # Only static
                members_final_df = members_static_df
        elif not members_dynamic_df.empty: # Only dynamic
            members_final_df = members_dynamic_df
    except Exception as e: print(f"Error merging members data: {e}")

    # Save final group/member files
    if not groups_final_df.empty:
        groups_final_df = optimize_dtypes(groups_final_df)
        try:
            groups_final_df.to_parquet(groups_final_path, engine='pyarrow', index=False)
            print(f"Saved final groups data to {groups_final_path}")
        except Exception as e: print(f"Error saving final groups data: {e}")
    else: print("No final groups data to save.")

    if not members_final_df.empty:
        members_final_df = optimize_dtypes(members_final_df)
        try:
            members_final_df.to_parquet(members_final_path, engine='pyarrow', index=False)
            print(f"Saved final members data to {members_final_path}")
        except Exception as e: print(f"Error saving final members data: {e}")
    else: print("No final members data to save.")

    # Final cleanup
    del groups_final_df, members_final_df, groups_dynamic_df, members_dynamic_df, groups_static_df, members_static_df
    del group_lookup, member_lookup
    gc.collect()

    print(f"\nData extraction process finished!")
    print(f"Output saved in: {output_dir}")
    print("Individual data types (feeds, messages, posts, events, etc.) are saved as chunked Parquet files.")
    print(f"Example: Read all group feeds using: pd.read_parquet('{os.path.join(output_dir, 'group_feeds_*.parquet')}')")
    print(f"Example: Read all post comments using: pd.read_parquet('{os.path.join(output_dir, 'post_comments_*.parquet')}')")
    print(f"Example: Read all member events using: pd.read_parquet('{os.path.join(output_dir, 'member_events_*.parquet')}')") # <<< Added example for events
    print("Consider using Dask or Spark for reading large partitioned datasets.")


# --- Main Execution Block ---
if __name__ == "__main__":
    # >>>>> IMPORTANT: SET YOUR DATA DIRECTORY HERE <<<<<
    data_dir = "Bunnings_v2" # <--- EXAMPLE: Set your *root* input directory

    if not os.path.isdir(data_dir):
        print(f"Error: Directory not found: {data_dir}")
        print("Please ensure the 'data_dir' variable points to the correct root location.")
    else:
        extract_data(data_dir)