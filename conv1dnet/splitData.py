import random

def split_data(filename, train_p = 0.8, val_p = 0.1):
    with open(filename, "r") as file:
        lines = file.readlines()
        n = len(lines)
        train_cnt = int(n * train_p)
        val_cnt = int(n * val_p)
        random.shuffle(lines)

        train_data = lines[:train_cnt]
        val_data = lines[train_cnt:train_cnt + val_cnt]
        test_data = lines[train_cnt + val_cnt:]

    return train_data, val_data, test_data

def write_to_file(filename, data):
    with open(filename, "w") as file:
        file.writelines(data)

if __name__ == "__main__":
    train, val, test = split_data("filtered-data.csv")        
    write_to_file("train-data.csv", train)
    write_to_file("val-data.csv", val)
    write_to_file("test-data.csv", test)