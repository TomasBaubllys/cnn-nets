import csv
import random

def filter_data(users_to_keep=30):
    users = []
    user_data_dict = {}
    user_count = 0
    with open('DSL-StrongPasswordData.csv') as file:
        csv_file = csv.reader(file)
        next(csv_file)
        for i, line in enumerate(csv_file):
            user = line[0]
            session = line[1]
            if user not in users and user_count < users_to_keep:
                users.append(user)
                user_data_dict[user + "_" + session] = [line]
                user_count += 1
            elif user + "_" + session in user_data_dict.keys():
                user_data_dict[user + "_" + session].append(line)

        return user_data_dict

def write_data(filename, data):
    with open(filename, "w") as file:
        csv_file = csv.writer(file)
        for i, values in enumerate(data.values()):
            for line in values:
                line[0] = i
                line.pop(1)
                line.pop(1)
                csv_file.writerow(line)

if __name__ == "__main__":
    data = filter_data(30)
    write_data("filtered-data.csv", data)
    #print(len(data["s003_1"]))
