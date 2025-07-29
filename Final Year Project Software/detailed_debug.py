import os

print("=== DETAILED FILE DEBUGGING ===")
print(f"Current working directory: {os.getcwd()}")
print(f"Script location: {os.path.dirname(os.path.abspath(__file__))}")
print()

# List ALL files in current directory
print("ALL files in current directory:")
all_files = os.listdir(".")
for i, file in enumerate(all_files, 1):
    print(f"{i:2d}. {file}")
print()

# Look specifically for CSV files
csv_files = [f for f in all_files if f.lower().endswith('.csv')]
print("CSV files found:")
if csv_files:
    for csv in csv_files:
        print(f"  📄 '{csv}'")
        print(f"     Full path: {os.path.abspath(csv)}")
        print(f"     File exists: {os.path.exists(csv)}")
        print(f"     File size: {os.path.getsize(csv) if os.path.exists(csv) else 'N/A'} bytes")
        print()
else:
    print("  ❌ No CSV files found!")
print()

# Check the exact filenames we're looking for
required_files = [
    "Train csv_version.csv",
    "Test csv_version.csv", 
    "Valid csv_version.csv"
]

print("Checking for EXACT required filenames:")
for filename in required_files:
    exists = os.path.exists(filename)
    print(f"  {'✅' if exists else '❌'} '{filename}' - Exists: {exists}")
    
    if not exists:
        # Look for similar files
        similar = [f for f in csv_files if 'train' in f.lower() or 'test' in f.lower() or 'valid' in f.lower()]
        if similar:
            print(f"      Similar files found: {similar}")
print()

# Test the actual paths that getData() would use
print("Testing paths that getData('.') would generate:")
data_dir = "."
for filename in required_files:
    full_path = os.path.join(data_dir, filename)
    exists = os.path.exists(full_path)
    print(f"  {'✅' if exists else '❌'} {full_path} - Exists: {exists}")

print()
print("=== NEXT STEPS ===")
if not any(os.path.exists(f) for f in required_files):
    print("❌ Required files not found with exact names.")
    print("🔧 Possible solutions:")
    print("   1. Check if your CSV files have slightly different names")
    print("   2. Rename your files to match exactly:")
    for filename in required_files:
        print(f"      - {filename}")
    print("   3. Or modify the code to use your actual filenames")
else:
    print("✅ Files found! The issue might be elsewhere.")