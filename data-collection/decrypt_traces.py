import os
import zipfile
import sys
import json
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

DEFAULT_PASSWORD = "hal1234"

def _derive_key(password: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=480000)
    return base64.urlsafe_b64encode(kdf.derive(password.encode()))

def decrypt_token_bytes(encrypted_data_b64: str, salt_b64: str, password: str = DEFAULT_PASSWORD) -> bytes:
    ct = base64.b64decode(encrypted_data_b64)
    salt = base64.b64decode(salt_b64)
    f = Fernet(_derive_key(password, salt))
    return f.decrypt(ct)

def decrypt_file_content(file_path):
    try:
        with open(file_path, 'r') as f:
            container = json.load(f)
        
        if "encrypted_data" not in container or "salt" not in container:
            print(f"Skipping {file_path}: Not a valid encrypted container.")
            return False

        plaintext = decrypt_token_bytes(container["encrypted_data"], container["salt"])
        
        output_path = file_path.replace('.encrypted', '')
        # Ensure we don't overwrite if it's the same name (though .encrypted implies it's not)
        if output_path == file_path:
             output_path += ".decrypted"

        with open(output_path, 'wb') as f:
            f.write(plaintext)
            
        print(f"Decrypted: {file_path} -> {output_path}")
        return True
    except Exception as e:
        print(f"Error decrypting {file_path}: {e}")
        return False

def process_traces(root_dir):
    # We repeatedly walk the directory to handle nested zips or just ensuring we catch everything
    # However, the structure seems flat: traces/*.zip -> *.json.encrypted
    
    # First pass: Unzip all zip files
    # We collect list first to avoid modifying list while iterating if we were doing something else, 
    # but os.walk is a generator.
    
    # We'll do a robust loop: keep looking for zips until no more zips are found
    # (Just in case zips contain zips, though unlikely here)
    # The requirement is just to unzip the ones in traces.
    
    print("--- Phase 1: Unzipping ---")
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.zip'):
                zip_path = os.path.join(root, file)
                try:
                    print(f"Processing Zip: {zip_path}")
                    with zipfile.ZipFile(zip_path, 'r') as zf:
                        zf.extractall(path=root, pwd=DEFAULT_PASSWORD.encode())
                    
                    # Delete zip file after successful extraction
                    os.remove(zip_path)
                    print(f"Extracted and deleted: {zip_path}")
                except Exception as e:
                    print(f"Error handling zip {zip_path}: {e}")

    print("\n--- Phase 2: Decrypting .encrypted files ---")
    # Second pass: Decrypt all .encrypted files
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.encrypted'):
                file_path = os.path.join(root, file)
                if decrypt_file_content(file_path):
                    # Delete the encrypted file after successful decryption
                    try:
                        os.remove(file_path)
                        print(f"Deleted encrypted file: {file_path}")
                    except Exception as e:
                        print(f"Error deleting {file_path}: {e}")

if __name__ == "__main__":
    # Check for traces folder in current dir or parent dir (relative to where script is run)
    # The script is in data-collection/, so traces should be in ../traces
    
    # We need to determine the absolute path of the traces directory
    # If run from root: traces/
    # If run from data-collection/: ../traces/
    
    possible_paths = [
        os.path.join(os.getcwd(), 'traces'),
        os.path.join(os.path.dirname(os.getcwd()), 'traces'), # if cwd is data-collection
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'traces')) # relative to script location
    ]
    
    traces_dir = None
    for p in possible_paths:
        if os.path.isdir(p):
            traces_dir = p
            break
            
    if not traces_dir:
        # Fallback: check if we are in root and traces exists
        if os.path.isdir("traces"):
             traces_dir = "traces"
        else:
            print("Directory 'traces' not found.")
            sys.exit(1)
        
    print(f"Target traces directory: {traces_dir}")
    process_traces(traces_dir)
    print("Process completed.")
