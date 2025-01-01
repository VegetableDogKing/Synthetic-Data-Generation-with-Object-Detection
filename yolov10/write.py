import os

for i in ["train", "val", "test"]:
    # Specify the directory you want to scan
    target_directory = f'yolov10/datasets/images/{i}'

    # Specify the output text file
    output_file = f'yolov10/datasets/{i}.txt'

    # Open the output file in write mode
    with open(output_file, 'w') as f:
        # Traverse the directory and its subdirectories
        for root, dirs, files in os.walk(target_directory):
            for file in files:
                # Get the relative path of the file
                relative_path = os.path.relpath(os.path.join(root, file), target_directory)
                # Write the relative path to the text file
                f.write(f'./images/{i}/' + relative_path + '\n')

    print(f'All file paths have been written to {output_file}')
