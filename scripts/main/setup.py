import os
import tarfile
import argparse
from tqdm import tqdm

def extract_tar_files(tar_files_dir, json_output_dir, jpg_output_dir, start_number, end_number):
    os.makedirs(json_output_dir, exist_ok=True)
    os.makedirs(jpg_output_dir, exist_ok=True)

    for tar_file_name in tqdm(os.listdir(tar_files_dir)):
        if tar_file_name.endswith(".tar"):
            try:
                tar_file_number = int(tar_file_name.split("_")[1].split(".")[0])
            except ValueError:
                continue
            
            if start_number <= tar_file_number < end_number:
                tar_file_path = os.path.join(tar_files_dir, tar_file_name)
                with tarfile.open(tar_file_path, "r") as tar:
                    for member in tar.getmembers():
                        if member.isfile():
                            file_name = os.path.basename(member.name)
                            if member.name.endswith(".json"):
                                member.name = file_name
                                tar.extract(member, json_output_dir)
                            elif member.name.endswith(".jpg"):
                                member.name = file_name
                                tar.extract(member, jpg_output_dir)

    print("Extraction completed")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory containing tar files")
    parser.add_argument("--json_output_dir", required=True, help="Directory to save json files")
    parser.add_argument("--jpg_output_dir", required=True, help="Directory to save jpg files")
    parser.add_argument("--start_number", type=int, default=0, help="First index of tar files to extract")
    parser.add_argument("--end_number", type=int, default=1000, help="Last index of tar files to extract. This number is not included.")

    args = parser.parse_args()

    extract_tar_files(args.input_dir, args.json_output_dir, args.jpg_output_dir, args.start_number, args.end_number)

if __name__ == "__main__":
    main()
