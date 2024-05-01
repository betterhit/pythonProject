import csv


def filter_and_write(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile, delimiter=',')
        writer = csv.writer(outfile, delimiter=',')
        print(reader)
        for row in reader:
            print(row)
            first_column_parts = row[0].split('-')

            filtered_cols = [col for col in row[1:] if col.split('-')[0] != first_column_parts[0]]
            print(filtered_cols)
            # 将第一列和筛选后的列写入新的CSV文件
            writer.writerow([row[0]] + filtered_cols)


# 调用函数
filter_and_write("output_sorted_keys.csv", "filter_output.csv")
