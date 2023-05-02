import subprocess
import time

# replace "commands.txt" with the name of your text file
with open("test_command.txt", "r") as f:
    command_lines = f.read().split("\n\n")

for ii in range(20):
# split the command blocks into pairs of commands
    command_liness = [c.strip().format(test_num=ii) for c in command_lines]

    command_pairs = [command_liness[i:i+2] for i in range(0, len(command_lines), 2)]

    for command_pair in command_pairs:
        # concatenate the two commands with a semicolon
        command = ";".join(command_pair)

        # run the command and capture its output
        output = subprocess.check_output(command, shell=True)

        # print the output
        print(output.decode())

        # wait for 5 minutes before running the next command
        time.sleep(60)
