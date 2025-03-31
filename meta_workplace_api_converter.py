import pandas as pd
import json
from pandas import json_normalize
import os
import warnings
import glob
from tqdm import tqdm

warnings.filterwarnings("ignore")

def process_json_group_data(json_data, groups_only_df, group_lookup=None):
    """
    Process JSON data and extract members IDs, feed IDs, and feed information.
    
    Args:
        json_data (dict): The JSON data to process
        groups_only_df (DataFrame): DataFrame containing group information with 'id' column
        group_lookup (dict, optional): Existing lookup dictionary for group data
        
    Returns:
        tuple: (updated_groups_df, feed_df, group_lookup) - DataFrames and updated lookup
    """
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
    
    # Process each group in the JSON
    for group_id, group_data in json_data.items():
        
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

    # Convert the lookup dictionary back to a DataFrame
    updated_groups_df = pd.DataFrame.from_dict(group_lookup, orient='index').reset_index()
    updated_groups_df.rename(columns={'index': 'id'}, inplace=True)  # Rename to 'id' to match original
    
    return updated_groups_df, feed_df, group_lookup

def process_json_member_conversation_data(json_data, member_only_df, member_lookup=None):
    """
    Process JSON data for member conversations and extract message information.
    
    Args:
        json_data (dict): The JSON data to process with members and their conversations
        member_only_df (DataFrame): DataFrame containing member information with 'id' column
        member_lookup (dict, optional): Existing lookup dictionary for member data
        
    Returns:
        tuple: (updated_members_df, messages_df, member_lookup) - DataFrames and updated lookup
    """
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
    
    # Process each member in the JSON
    for member_id, member_data in json_data.items():
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

    # Convert the lookup dictionary back to a DataFrame
    updated_members_df = pd.DataFrame.from_dict(member_lookup, orient='index').reset_index()
    updated_members_df.rename(columns={'index': 'id'}, inplace=True)
    
    return updated_members_df, messages_df, member_lookup

def process_json_member_data(json_data, member_only_df, member_lookup=None):
    """
    Process JSON data for member information, including feeds, conversations, managers, and groups.
    
    Args:
        json_data (dict): The JSON data to process with member information
        member_only_df (DataFrame): DataFrame containing member information with 'id' column
        member_lookup (dict, optional): Existing lookup dictionary for member data
        
    Returns:
        tuple: (updated_members_df, member_feeds_df, conversations_df, member_lookup)
    """
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
    
    # Process each member in the JSON
    for member_id, member_data in json_data.items():
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
            member_lookup[member_id]['conversation_ids'] = list(existing_conv_ids_set) or None
            member_lookup[member_id]['feed_ids'] = feed_ids or None
            member_lookup[member_id]['manager_ids'] = manager_ids or None
            member_lookup[member_id]['group_ids'] = group_ids or None
        else:
            # Create a new entry if the member is not in the member_only_df
            member_lookup[member_id] = {
                'conversation_ids': conversation_ids or None,
                'feed_ids': feed_ids or None,
                'manager_ids': manager_ids or None,
                'group_ids': group_ids or None
            }

    # Convert the lookup dictionary back to a DataFrame
    updated_members_df = pd.DataFrame.from_dict(member_lookup, orient='index').reset_index()
    updated_members_df.rename(columns={'index': 'id'}, inplace=True)
    
    return updated_members_df, member_feeds_df, conversations_df, member_lookup

def extract_data(data_dir):
    """
    Extract and process data from a directory structure containing JSON files.
    
    Args:
        data_dir (str): Path to the directory containing the data
        
    Returns:
        tuple: (groups_df, members_df, group_feeds_df, member_feeds_df, 
                conversations_df, messages_df) - Processed DataFrames
    """
    # Initialize empty DataFrames for collecting results
    all_group_feeds_df = pd.DataFrame()
    all_member_feeds_df = pd.DataFrame()
    all_conversations_df = pd.DataFrame()
    all_messages_df = pd.DataFrame()
    
    # Path to groups and members JSON files
    groups_only_data_path = os.path.join(data_dir, 'groups.json')
    members_only_data_path = os.path.join(data_dir, 'members.json')
    
    # Read and process groups data
    print("Reading groups data...")
    with open(groups_only_data_path) as f:
        groups_only_data = json.load(f)
    groups_only_df = pd.json_normalize(groups_only_data)

    # Process each member type separately and add the member_type column
    members_dfs = []
    
    # # Read and process members data
    print("Reading members data...")
    with open(members_only_data_path) as f:
        members_only_data = json.load(f)

    # Process admin members first (highest priority)
    if "admins" in members_only_data and isinstance(members_only_data["admins"], list):
        admin_members_df = pd.json_normalize(members_only_data["admins"])
        admin_members_df["member_type"] = "admin"
        members_dfs.append(admin_members_df)
        # Keep track of admin IDs to filter out duplicates later
        admin_ids = set(admin_members_df['id'].tolist())
    else:
        admin_ids = set()

    # Process current members (excluding those who are admins)
    if "members" in members_only_data and isinstance(members_only_data["members"], list):
        current_members_df = pd.json_normalize(members_only_data["members"])
        # Filter out members who are also admins
        if admin_ids:
            current_members_df = current_members_df[~current_members_df['id'].isin(admin_ids)]
        current_members_df["member_type"] = "current"
        members_dfs.append(current_members_df)
        # Update set of processed IDs
        processed_ids = admin_ids.union(set(current_members_df['id'].tolist()))
    else:
        processed_ids = admin_ids

    # Process former members (excluding duplicates)
    if "former_members" in members_only_data and isinstance(members_only_data["former_members"], list):
        former_members_df = pd.json_normalize(members_only_data["former_members"])
        # Filter out already processed members
        if processed_ids:
            former_members_df = former_members_df[~former_members_df['id'].isin(processed_ids)]
        former_members_df["member_type"] = "former"
        members_dfs.append(former_members_df)

    # Combine all member DataFrames
    members_only_df = pd.concat(members_dfs, ignore_index=True) if members_dfs else pd.DataFrame()
        
    # Create lookup dictionaries
    group_lookup = None
    member_lookup = None
    
    # Find all JSON files in the directory, excluding groups.json and members.json
    all_json_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.json') and file not in ['groups.json', 'members.json']:
                all_json_files.append(os.path.join(root, file))
    
    print(f"Found {len(all_json_files)} JSON files to process.")
    
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
        
        # Load the JSON file
        with open(file_path) as f:
            json_data = json.load(f)
        
        # Process based on the directory name
        if parent_dir == 'group-data':
            # Process group data
            updated_groups_df, group_feeds_df, group_lookup = process_json_group_data(
                json_data, groups_only_df, group_lookup
            )
            
            # Update groups DataFrame for next iteration
            groups_only_df = updated_groups_df
            
            # Accumulate feed data
            if not group_feeds_df.empty:
                all_group_feeds_df = pd.concat([all_group_feeds_df, group_feeds_df], ignore_index=True)
                
        elif parent_dir == 'member-data':
            # Process member data
            updated_members_df, member_feeds_df, conversations_df, member_lookup = process_json_member_data(
                json_data, members_only_df, member_lookup
            )
            
            # Update members DataFrame for next iteration
            members_only_df = updated_members_df
            
            # Accumulate feed and conversation data
            if not member_feeds_df.empty:
                all_member_feeds_df = pd.concat([all_member_feeds_df, member_feeds_df], ignore_index=True)
            
            if not conversations_df.empty:
                all_conversations_df = pd.concat([all_conversations_df, conversations_df], ignore_index=True)
                
        elif parent_dir == 'conversation-messages':
            # Process conversation messages
            updated_members_df, messages_df, member_lookup = process_json_member_conversation_data(
                json_data, members_only_df, member_lookup
            )
            
            # Update members DataFrame for next iteration
            members_only_df = updated_members_df
            
            # Accumulate message data
            if not messages_df.empty:
                all_messages_df = pd.concat([all_messages_df, messages_df], ignore_index=True)
        
        else:
            # Update progress bar with skipped file
            pbar.set_description(f"Skipping {parent_dir}/{file_name} - unknown directory type")
    
    # Close the progress bar
    pbar.close()
    
    print("Processing complete!")

    output_dir = data_dir + '_converted'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Saving data to {output_dir}...")

    # Save all DataFrames to PKL files
    groups_only_df.to_pickle(os.path.join(output_dir, 'groups.pkl'))
    members_only_df.to_pickle(os.path.join(output_dir, 'members.pkl'))
    all_group_feeds_df.to_pickle(os.path.join(output_dir, 'group_feeds.pkl'))
    all_member_feeds_df.to_pickle(os.path.join(output_dir, 'member_feeds.pkl'))
    all_conversations_df.to_pickle(os.path.join(output_dir, 'conversations.pkl'))
    all_messages_df.to_pickle(os.path.join(output_dir, 'member_messages.pkl'))

    # Save all DataFrames to CSV files
    groups_only_df.to_csv(os.path.join(output_dir, 'groups.csv'), index=False)
    members_only_df.to_csv(os.path.join(output_dir, 'members.csv'), index=False)
    all_group_feeds_df.to_csv(os.path.join(output_dir, 'group_feeds.csv'), index=False)
    all_member_feeds_df.to_csv(os.path.join(output_dir, 'member_feeds.csv'), index=False)
    all_conversations_df.to_csv(os.path.join(output_dir, 'conversations.csv'), index=False)
    all_messages_df.to_csv(os.path.join(output_dir, 'member_messages.csv'), index=False)


    print(f"Data successfully saved to {output_dir}!")
    
    # Return all the processed DataFrames
    return (
        groups_only_df,           # Updated groups DataFrame
        members_only_df,          # Updated members DataFrame
        all_group_feeds_df,       # Group feeds
        all_member_feeds_df,      # Member feeds
        all_conversations_df,     # Conversations
        all_messages_df           # Messages
    )

if __name__ == "__main__":
    data_dir = "Bunnings"
    groups_df, members_df, group_feeds_df, member_feeds_df, conversations_df, messages_df = extract_data(data_dir)
    print("Data extraction complete!")