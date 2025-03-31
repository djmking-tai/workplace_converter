import pandas as pd
import json
from pandas import json_normalize
import os
import warnings
import glob
import gc
import uuid
from tqdm import tqdm
import tempfile

warnings.filterwarnings("ignore")

def read_json_in_batches(file_path, batch_size=100):
    """Read a JSON file in batches to reduce memory usage."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Process dictionary in batches
    batch = {}
    count = 0
    
    for key, value in data.items():
        batch[key] = value
        count += 1
        
        if count >= batch_size:
            yield batch
            batch = {}
            count = 0
            
    # Yield the last batch if it exists
    if batch:
        yield batch
    
    # Free memory
    del data
    gc.collect()

def save_df_chunk(df, output_dir, prefix, idx):
    """Save a DataFrame chunk to disk and return the filename."""
    if df.empty:
        return None
    
    # Ensure temp directory exists
    temp_dir = os.path.join(output_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Save to disk
    filename = f"{prefix}_{idx}.pkl"
    filepath = os.path.join(temp_dir, filename)
    df.to_pickle(filepath)
    
    # Clear memory
    del df
    gc.collect()
    
    return filepath

def combine_chunk_files(output_dir, prefix, output_filename):
    """Combine chunked files into a single DataFrame and save."""
    # Find all chunk files
    chunk_pattern = os.path.join(output_dir, "temp", f"{prefix}_*.pkl")
    chunk_files = glob.glob(chunk_pattern)
    
    if not chunk_files:
        print(f"No chunks found for {prefix}")
        return pd.DataFrame()
    
    print(f"Combining {len(chunk_files)} chunks for {prefix}...")
    
    # Create an empty DataFrame to store the combined data
    combined_df = pd.DataFrame()
    
    # Process chunks one at a time
    for chunk_file in tqdm(chunk_files, desc=f"Combining {prefix} chunks"):
        # Read chunk
        chunk_df = pd.read_pickle(chunk_file)
        
        # Append to combined DataFrame
        combined_df = pd.concat([combined_df, chunk_df], ignore_index=True)
        
        # Free memory
        del chunk_df
        gc.collect()
        
        # Remove chunk file after processing
        os.remove(chunk_file)
    
    # Save combined DataFrame
    output_path = os.path.join(output_dir, output_filename)
    combined_df.to_pickle(output_path)
    print(f"Saved combined {prefix} data to {output_path}")
    
    # Also save as CSV
    csv_path = output_path.replace('.pkl', '.csv')
    combined_df.to_csv(csv_path, index=False)
    
    return combined_df

def process_group_batch(batch_data, groups_only_df, group_lookup=None):
    """Process a batch of group data and return feed items and updated lookup."""
    # Initialize empty feed DataFrame
    feed_df = pd.DataFrame()
    
    # Use provided lookup or create a new one if not provided
    if group_lookup is None:
        # Create a copy of the groups_only_df to avoid modifying the original
        groups_df = groups_only_df.copy()
        
        # Create columns for the member lists if they don't exist
        for col in ['current_members', 'admin_members', 'former_members', 'feed_ids']:
            if col not in groups_df.columns:
                groups_df[col] = None
        
        # Create a lookup dictionary for faster access (O(1) instead of O(n))
        group_lookup = groups_df.set_index('id').to_dict(orient='index')
    
    # Process each group in the batch
    for group_id, group_data in batch_data.items():
        # Extract member IDs
        current_members = []
        admin_members = []
        former_members = []
        feed_ids = []
        
        # Process current members
        if 'members' in group_data and isinstance(group_data['members'], list):
            current_members = [item['id'] for item in group_data['members'] 
                              if isinstance(item, dict) and 'id' in item]
            
        # Process admin members
        if 'admins' in group_data and isinstance(group_data['admins'], list):
            admin_members = [item['id'] for item in group_data['admins'] 
                            if isinstance(item, dict) and 'id' in item]
            
        # Process former members
        if 'former_members' in group_data and isinstance(group_data['former_members'], list):
            former_members = [item['id'] for item in group_data['former_members'] 
                             if isinstance(item, dict) and 'id' in item]
        
        # Process feed items
        if 'feed' in group_data and isinstance(group_data['feed'], list):
            feed_items = [item for item in group_data['feed'] 
                         if isinstance(item, dict) and 'id' in item]
            
            # Extract feed IDs
            feed_ids = [item['id'] for item in feed_items if 'id' in item]
            
            # Pre-process feed items to handle the properties field
            for item in feed_items:
                if 'properties' in item and isinstance(item['properties'], list) and len(item['properties']) > 0:
                    # Extract the first (and typically only) properties item
                    props = item['properties'][0]
                    if isinstance(props, dict):
                        # Convert properties to flat fields
                        for key, value in props.items():
                            item[f'properties_{key}'] = value
                    # Remove the original properties list
                    del item['properties']
                    
            if feed_items:
                # Use json_normalize
                group_feed_df = pd.json_normalize(feed_items, sep='_')
                group_feed_df['group_id'] = group_id
                feed_df = pd.concat([feed_df, group_feed_df], ignore_index=True)
        
        # Update the group information in the lookup dictionary
        if group_id in group_lookup:
            group_lookup[group_id]['current_members'] = current_members or None
            group_lookup[group_id]['admin_members'] = admin_members or None
            group_lookup[group_id]['former_members'] = former_members or None
            group_lookup[group_id]['feed_ids'] = feed_ids or None
        else:
            # Create a new entry if the group is not in the groups_only_df
            group_lookup[group_id] = {
                'current_members': current_members or None,
                'admin_members': admin_members or None,
                'former_members': former_members or None,
                'feed_ids': feed_ids or None
            }
    
    return feed_df, group_lookup

def process_member_batch(batch_data, member_only_df, member_lookup=None):
    """Process a batch of member data and return feed items, conversation data, and updated lookup."""
    # Initialize empty DataFrames
    member_feeds_df = pd.DataFrame()
    conversations_df = pd.DataFrame()
    
    # Use provided lookup or create a new one if not provided
    if member_lookup is None:
        # Create a copy of the member_only_df to avoid modifying the original
        members_df = member_only_df.copy()
        
        # Create columns if they don't exist
        for col in ['conversation_ids', 'feed_ids', 'manager_ids', 'group_ids']:
            if col not in members_df.columns:
                members_df[col] = None
        
        # Create a lookup dictionary for faster access (O(1) instead of O(n))
        member_lookup = members_df.set_index('id').to_dict(orient='index')
    
    # Process each member in the batch
    for member_id, member_data in batch_data.items():
        # Initialize lists to store IDs
        feed_ids = []
        manager_ids = []
        group_ids = []

        # Get existing conversation IDs from lookup or initialize as empty list
        if member_id in member_lookup and 'conversation_ids' in member_lookup[member_id]:
            # Get existing IDs, handle None case
            existing_conv_ids = member_lookup[member_id]['conversation_ids'] or []
            # Convert to list if it's not already
            if not isinstance(existing_conv_ids, list):
                existing_conv_ids = [existing_conv_ids]
        else:
            existing_conv_ids = []

        existing_conv_ids_set = set(existing_conv_ids)
        
        # Process feed items
        if 'feed' in member_data and isinstance(member_data['feed'], list):
            feed_items = [item for item in member_data['feed'] 
                         if isinstance(item, dict) and 'id' in item]
            
            # Extract feed IDs
            feed_ids = [item['id'] for item in feed_items if 'id' in item]
            
            # Pre-process feed items to handle any nested structures if needed
            for item in feed_items:
                if 'properties' in item and isinstance(item['properties'], list) and len(item['properties']) > 0:
                    # Extract the first (and typically only) properties item
                    props = item['properties'][0]
                    if isinstance(props, dict):
                        # Convert properties to flat fields
                        for key, value in props.items():
                            item[f'properties_{key}'] = value
                    # Remove the original properties list
                    del item['properties']
                    
            if feed_items:
                # Use json_normalize
                member_feed_df = pd.json_normalize(feed_items, sep='_')
                member_feed_df['member_id'] = member_id  # Associate with member instead of group
                member_feeds_df = pd.concat([member_feeds_df, member_feed_df], ignore_index=True)
        
        # Process conversations
        if 'conversations' in member_data and isinstance(member_data['conversations'], list):
            conversations = [conv for conv in member_data['conversations'] 
                            if isinstance(conv, dict) and 'id' in conv]
            
            # Extract conversation IDs
            conversation_ids = [conv['id'] for conv in conversations if 'id' in conv]
            existing_conv_ids_set.update(conversation_ids)
            
            # Process each conversation
            for conv in conversations:
                # Extract participant IDs if they exist
                participant_ids = []
                if ('participants' in conv and isinstance(conv['participants'], dict) and 
                    'data' in conv['participants'] and isinstance(conv['participants']['data'], list)):
                    participant_ids = [p.get('id') for p in conv['participants']['data'] 
                                      if isinstance(p, dict) and 'id' in p]
                
                # Create a clean conversation record
                conversation_record = {
                    'id': conv.get('id'),
                    'member_id': member_id,
                    'participant_ids': participant_ids,
                    'message_count': conv.get('message_count'),
                    'unread_count': conv.get('unread_count'),
                    'updated_time': conv.get('updated_time')
                }
                
                # Add to conversations DataFrame
                conversations_df = pd.concat([conversations_df, 
                                             pd.DataFrame([conversation_record])], 
                                            ignore_index=True)
        
        # Process managers
        if 'managers' in member_data and isinstance(member_data['managers'], list):
            managers = [m for m in member_data['managers'] 
                       if isinstance(m, dict) and 'id' in m]
            manager_ids = [m['id'] for m in managers if 'id' in m]
        
        # Process groups
        if 'groups' in member_data and isinstance(member_data['groups'], list):
            groups = [g for g in member_data['groups'] 
                     if isinstance(g, dict) and 'id' in g]
            group_ids = [g['id'] for g in groups if 'id' in g]
        
        # Update the member information in the lookup dictionary
        if member_id in member_lookup:
            member_lookup[member_id]['conversation_ids'] = list(existing_conv_ids_set) if existing_conv_ids_set else None
            member_lookup[member_id]['feed_ids'] = feed_ids or None
            member_lookup[member_id]['manager_ids'] = manager_ids or None
            member_lookup[member_id]['group_ids'] = group_ids or None
        else:
            # Create a new entry if the member is not in the member_only_df
            member_lookup[member_id] = {
                'conversation_ids': list(existing_conv_ids_set) if existing_conv_ids_set else None,
                'feed_ids': feed_ids or None,
                'manager_ids': manager_ids or None,
                'group_ids': group_ids or None
            }
    
    return member_feeds_df, conversations_df, member_lookup

def process_message_batch(batch_data, member_only_df, member_lookup=None):
    """Process a batch of member conversation messages and return message data and updated lookup."""
    # Initialize empty messages DataFrame
    messages_df = pd.DataFrame()
    
    # Use provided lookup or create a new one if not provided
    if member_lookup is None:
        # Create a copy of the member_only_df to avoid modifying the original
        members_df = member_only_df.copy()
        
        # Create columns if they don't exist - include all columns used in both functions
        for col in ['conversation_ids', 'feed_ids', 'manager_ids', 'group_ids']:
            if col not in members_df.columns:
                members_df[col] = None
        
        # Create a lookup dictionary for faster access (O(1) instead of O(n))
        member_lookup = members_df.set_index('id').to_dict(orient='index')
    
    # Process each member in the batch
    for member_id, member_data in batch_data.items():
        # Get existing conversation IDs from lookup or initialize as empty list
        if member_id in member_lookup and 'conversation_ids' in member_lookup[member_id]:
            # Get existing IDs, handle None case
            existing_conv_ids = member_lookup[member_id]['conversation_ids'] or []
            # Convert to list if it's not already
            if not isinstance(existing_conv_ids, list):
                existing_conv_ids = [existing_conv_ids]
        else:
            existing_conv_ids = []

        # New conversation IDs from this data part
        existing_conv_ids_set = set(existing_conv_ids)
        
        # Process each conversation
        for conversation_id, messages in member_data.items():
            # Add conversation ID to the list
            existing_conv_ids_set.add(conversation_id)
            
            # Process messages if they exist and are in a list
            if isinstance(messages, list) and len(messages) > 0:
                # Process message tags
                for message in messages:
                    # Handle tags if present
                    if 'tags' in message and isinstance(message['tags'], dict) and 'data' in message['tags']:
                        # Extract tag names into a list
                        tag_data = message['tags']['data']
                        if isinstance(tag_data, list):
                            tag_names = [item.get('name') for item in tag_data if isinstance(item, dict) and 'name' in item]
                            message['tag_names'] = tag_names
                        # Remove the original tags structure
                        del message['tags']
                
                # Normalize the messages to a DataFrame
                try:
                    conversation_messages_df = pd.json_normalize(messages, sep='_')
                    
                    # Add member and conversation IDs
                    conversation_messages_df['member_id'] = member_id
                    conversation_messages_df['conversation_id'] = conversation_id
                    
                    # Extract community ID if it exists
                    if 'from_community_id' in conversation_messages_df.columns:
                        # Already flattened by json_normalize
                        pass
                    elif any('from.community.id' in col for col in conversation_messages_df.columns):
                        # Rename the column to a simpler format
                        cols_to_rename = {col: 'from_community_id' 
                                          for col in conversation_messages_df.columns 
                                          if col == 'from.community.id'}
                        conversation_messages_df.rename(columns=cols_to_rename, inplace=True)
                    
                    # Append to the messages DataFrame
                    messages_df = pd.concat([messages_df, conversation_messages_df], ignore_index=True)
                except Exception as e:
                    print(f"Error processing conversation {conversation_id} for member {member_id}: {e}")
        
        # Update the member information in the lookup dictionary
        if member_id in member_lookup:
            # Update the conversation_ids field with our merged list
            member_lookup[member_id]['conversation_ids'] = list(existing_conv_ids_set) if existing_conv_ids_set else None
        else:
            # Create a new entry with ALL columns initialized
            member_lookup[member_id] = {
                'conversation_ids': list(existing_conv_ids_set) if existing_conv_ids_set else None,
                'feed_ids': None,
                'manager_ids': None,
                'group_ids': None
            }
    
    return messages_df, member_lookup

def extract_data_optimized(data_dir):
    """
    Extract and process data with optimized memory usage.
    
    Args:
        data_dir (str): Path to the directory containing the data
    """
    # Create output directory
    output_dir = data_dir + '_converted'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "temp"), exist_ok=True)
    
    # Path to groups and members JSON files
    groups_only_data_path = os.path.join(data_dir, 'groups.json')
    members_only_data_path = os.path.join(data_dir, 'members.json')
    
    # Read and process groups data
    print("Reading groups data...")
    with open(groups_only_data_path) as f:
        groups_only_data = json.load(f)
    
    # Convert to DataFrame
    groups_only_df = pd.json_normalize(groups_only_data)
    
    # Free memory
    del groups_only_data
    gc.collect()
    
    # Prepare for member types
    print("Reading members data...")
    with open(members_only_data_path) as f:
        members_only_data = json.load(f)
    
    # Process each member type separately and add the member_type column
    members_dfs = []
    
    # Process admin members first (highest priority)
    admin_ids = set()
    if "admins" in members_only_data and isinstance(members_only_data["admins"], list):
        admin_members_df = pd.json_normalize(members_only_data["admins"])
        admin_members_df["member_type"] = "admin"
        members_dfs.append(admin_members_df)
        # Keep track of admin IDs to filter out duplicates later
        admin_ids = set(admin_members_df['id'].tolist())
        # Free memory
        del admin_members_df
        gc.collect()
    
    # Process current members (excluding those who are admins)
    processed_ids = admin_ids.copy()
    if "members" in members_only_data and isinstance(members_only_data["members"], list):
        current_members_df = pd.json_normalize(members_only_data["members"])
        # Filter out members who are also admins
        if admin_ids:
            current_members_df = current_members_df[~current_members_df['id'].isin(admin_ids)]
        current_members_df["member_type"] = "current"
        members_dfs.append(current_members_df)
        # Update set of processed IDs
        processed_ids.update(set(current_members_df['id'].tolist()))
        # Free memory
        del current_members_df
        gc.collect()
    
    # Process former members (excluding duplicates)
    if "former_members" in members_only_data and isinstance(members_only_data["former_members"], list):
        former_members_df = pd.json_normalize(members_only_data["former_members"])
        # Filter out already processed members
        if processed_ids:
            former_members_df = former_members_df[~former_members_df['id'].isin(processed_ids)]
        former_members_df["member_type"] = "former"
        members_dfs.append(former_members_df)
        # Free memory
        del former_members_df
        gc.collect()
    
    # Combine all member DataFrames
    print("Combining member DataFrames...")
    members_only_df = pd.DataFrame()
    for df in members_dfs:
        members_only_df = pd.concat([members_only_df, df], ignore_index=True)
        # Free memory after concatenation
        del df
        gc.collect()
    
    # Free memory
    del members_only_data, members_dfs, admin_ids, processed_ids
    gc.collect()
    
    # Create lookup dictionaries
    # This is done once per extraction to maintain state across files
    print("Creating lookup dictionaries...")
    
    # Create columns for the member lists if they don't exist
    for col in ['current_members', 'admin_members', 'former_members', 'feed_ids']:
        if col not in groups_only_df.columns:
            groups_only_df[col] = None
    
    # Create columns for members if they don't exist
    for col in ['conversation_ids', 'feed_ids', 'manager_ids', 'group_ids']:
        if col not in members_only_df.columns:
            members_only_df[col] = None
    
    # Create lookup dictionaries
    group_lookup = groups_only_df.set_index('id').to_dict(orient='index')
    member_lookup = members_only_df.set_index('id').to_dict(orient='index')
    
    # Save initial DataFrames to have something in case of failure
    print("Saving initial DataFrames...")
    groups_only_df.to_pickle(os.path.join(output_dir, 'groups_initial.pkl'))
    members_only_df.to_pickle(os.path.join(output_dir, 'members_initial.pkl'))
    
    # Find all JSON files in the directory, excluding groups.json and members.json
    all_json_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.json') and file not in ['groups.json', 'members.json']:
                all_json_files.append(os.path.join(root, file))
    
    print(f"Found {len(all_json_files)} JSON files to process.")
    
    # Initialize chunk counters
    group_feed_chunk_count = 0
    member_feed_chunk_count = 0
    conversation_chunk_count = 0
    message_chunk_count = 0
    
    # Maximum rows per chunk (adjust based on memory availability)
    max_rows_per_chunk = 50000
    
    # Batch size for reading JSON files (adjust based on memory availability)
    batch_size = 100
    
    # Create a progress bar for processing files
    pbar = tqdm(all_json_files, desc="Processing files")
    
    # Process each JSON file based on its directory name
    for file_path in pbar:
        # Get the parent directory name
        parent_dir = os.path.basename(os.path.dirname(file_path))
        file_name = os.path.basename(file_path)
        
        # For nested folders, get the parent's parent directory
        if parent_dir == 'conversation-messages':
            grand_parent_dir = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
            if grand_parent_dir == 'member-data':
                parent_dir = 'conversation-messages'
        
        # Update progress bar description with current file and its folder
        pbar.set_description(f"Processing {parent_dir}/{file_name}")
        
        try:
            # Initialize temporary DataFrames for batching
            temp_feed_df = pd.DataFrame()
            temp_conversations_df = pd.DataFrame()
            temp_messages_df = pd.DataFrame()
            
            # Process based on the directory name
            if parent_dir == 'group-data':
                # Process in batches to reduce memory usage
                for batch_data in read_json_in_batches(file_path, batch_size):
                    feed_df, group_lookup = process_group_batch(batch_data, groups_only_df, group_lookup)
                    
                    # Concatenate with temporary DataFrame
                    if not feed_df.empty:
                        temp_feed_df = pd.concat([temp_feed_df, feed_df], ignore_index=True)
                    
                    # Save chunk if it exceeds threshold
                    if len(temp_feed_df) >= max_rows_per_chunk:
                        save_df_chunk(temp_feed_df, output_dir, "group_feeds", group_feed_chunk_count)
                        group_feed_chunk_count += 1
                        temp_feed_df = pd.DataFrame()
                    
                    # Free memory
                    del feed_df, batch_data
                    gc.collect()
                
                # Save any remaining data
                if not temp_feed_df.empty:
                    save_df_chunk(temp_feed_df, output_dir, "group_feeds", group_feed_chunk_count)
                    group_feed_chunk_count += 1
                    del temp_feed_df
                    gc.collect()
                
            elif parent_dir == 'member-data':
                # Process in batches
                for batch_data in read_json_in_batches(file_path, batch_size):
                    member_feeds_df, conversations_df, member_lookup = process_member_batch(
                        batch_data, members_only_df, member_lookup
                    )
                    
                    # Concatenate with temporary DataFrames
                    if not member_feeds_df.empty:
                        temp_feed_df = pd.concat([temp_feed_df, member_feeds_df], ignore_index=True)
                    
                    if not conversations_df.empty:
                        temp_conversations_df = pd.concat([temp_conversations_df, conversations_df], ignore_index=True)
                    
                    # Save chunks if they exceed threshold
                    if len(temp_feed_df) >= max_rows_per_chunk:
                        save_df_chunk(temp_feed_df, output_dir, "member_feeds", member_feed_chunk_count)
                        member_feed_chunk_count += 1
                        temp_feed_df = pd.DataFrame()
                    
                    if len(temp_conversations_df) >= max_rows_per_chunk:
                        save_df_chunk(temp_conversations_df, output_dir, "conversations", conversation_chunk_count)
                        conversation_chunk_count += 1
                        temp_conversations_df = pd.DataFrame()
                    
                    # Free memory
                    del member_feeds_df, conversations_df, batch_data
                    gc.collect()
                
                # Save any remaining data
                if not temp_feed_df.empty:
                    save_df_chunk(temp_feed_df, output_dir, "member_feeds", member_feed_chunk_count)
                    member_feed_chunk_count += 1
                    del temp_feed_df
                    gc.collect()
                
                if not temp_conversations_df.empty:
                    save_df_chunk(temp_conversations_df, output_dir, "conversations", conversation_chunk_count)
                    conversation_chunk_count += 1
                    del temp_conversations_df
                    gc.collect()
                
            elif parent_dir == 'conversation-messages':
                # Process in batches
                for batch_data in read_json_in_batches(file_path, batch_size):
                    messages_df, member_lookup = process_message_batch(
                        batch_data, members_only_df, member_lookup
                    )
                    
                    # Concatenate with temporary DataFrame
                    if not messages_df.empty:
                        temp_messages_df = pd.concat([temp_messages_df, messages_df], ignore_index=True)
                    
                    # Save chunk if it exceeds threshold
                    if len(temp_messages_df) >= max_rows_per_chunk:
                        save_df_chunk(temp_messages_df, output_dir, "messages", message_chunk_count)
                        message_chunk_count += 1
                        temp_messages_df = pd.DataFrame()
                    
                    # Free memory
                    del messages_df, batch_data
                    gc.collect()
                
                # Save any remaining data
                if not temp_messages_df.empty:
                    save_df_chunk(temp_messages_df, output_dir, "messages", message_chunk_count)
                    message_chunk_count += 1
                    del temp_messages_df
                    gc.collect()
            
            else:
                # Skip unknown directory types
                pbar.set_description(f"Skipping {parent_dir}/{file_name} - unknown directory type")
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            # Continue processing other files even if one fails
            continue
    
    # Close the progress bar
    pbar.close()
    
    # Update groups and members with collected data
    print("Updating groups and members with collected data...")
    updated_groups_df = pd.DataFrame.from_dict(group_lookup, orient='index').reset_index()
    updated_groups_df.rename(columns={'index': 'id'}, inplace=True)
    
    updated_members_df = pd.DataFrame.from_dict(member_lookup, orient='index').reset_index()
    updated_members_df.rename(columns={'index': 'id'}, inplace=True)
    
    # Free memory
    del group_lookup, member_lookup
    gc.collect()
    
    # Create combined DataFrames
    print("Combining chunks...")
    
    # Combine chunks for each DataFrame type
    all_group_feeds_df = combine_chunk_files(output_dir, "group_feeds", "group_feeds.pkl")
    all_member_feeds_df = combine_chunk_files(output_dir, "member_feeds", "member_feeds.pkl")
    all_conversations_df = combine_chunk_files(output_dir, "conversations", "conversations.pkl")
    all_messages_df = combine_chunk_files(output_dir, "messages", "member_messages.pkl")
    
    # Save final results
    print("Saving final results...")
    
    # Save updated DataFrames
    updated_groups_df.to_pickle(os.path.join(output_dir, 'groups.pkl'))
    updated_groups_df.to_csv(os.path.join(output_dir, 'groups.csv'), index=False)
    
    updated_members_df.to_pickle(os.path.join(output_dir, 'members.pkl'))
    updated_members_df.to_csv(os.path.join(output_dir, 'members.csv'), index=False)
    
    print(f"Data successfully saved to {output_dir}!")
    
    # Clean up temp directory
    temp_dir = os.path.join(output_dir, "temp")
    if os.path.exists(temp_dir):
        try:
            import shutil
            shutil.rmtree(temp_dir)
            print(f"Removed temporary directory: {temp_dir}")
        except Exception as e:
            print(f"Could not remove temporary directory: {e}")
    
    return (
        updated_groups_df,           # Updated groups DataFrame
        updated_members_df,          # Updated members DataFrame
        all_group_feeds_df,          # Group feeds
        all_member_feeds_df,         # Member feeds
        all_conversations_df,        # Conversations
        all_messages_df              # Messages
    )