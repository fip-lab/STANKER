# encoding=utf-8
import csv
import json
import xlrd
import pandas


def read_file(file):
    """Read a txt file then return its content as list."""
    with open(file, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
    return lines


def read_csv_file(file):
    """Read a csv file then return its data as list."""
    output = []
    with open(file, 'r', encoding='utf-8') as fp:
        csv_reader = csv.reader(fp)
        for item in csv_reader:
            output.append(item)
        return output


def pd_read_csv_file(file):
    """Read a csv file by pandas then return its data as list."""
    output = []
    csv_data = pandas.read_csv(file, header=None)
    for index in csv_data.index:
        output.append(csv_data.loc[index].values)
    return output


def read_tsv_file(file):
    """Read a tsv file then return its data as list."""
    csv.register_dialect('tsv', delimiter='\t')
    output = []
    with open(file, 'r', encoding='utf-8') as fp:
        tsv_reader = csv.reader(fp, 'tsv')
        for item in tsv_reader:
            output.append(item)
        return output


def read_json_file(file):
    """Read a json file then return its data."""
    with open(file, 'r', encoding='utf-8') as fp:
        dicts = json.load(fp)
        return dicts


def read_xlsx_file(file, sheet_index=0):
    """Read a xlsx file then return its table."""
    table = []
    work_book = xlrd.open_workbook(file)
    work_sheet = work_book.sheet_by_index(sheet_index)
    n_rows = work_sheet.nrows
    for i in range(n_rows):
        table.append(work_sheet.row_values(i))
    return table


def write_file(file, data):
    """Write list of content to a txt file."""
    with open(file, 'w', encoding='utf-8', newline='') as fp:
        for line in data:
            fp.write(line + '\n')


def write_csv_file(file, data):
    """Write list data to a csv file."""
    with open(file, 'w', encoding='utf-8', newline='') as fp:
        csv_writer = csv.writer(fp)
        for item in data:
            csv_writer.writerow(item)


def write_tsv_file(file, data):
    """Write list data to a tsv file."""
    csv.register_dialect('tsv', delimiter='\t')
    with open(file, 'w', encoding='utf-8', newline='') as fp:
        tsv_writer = csv.writer(fp, 'tsv')
        for item in data:
            tsv_writer.writerow(item)


def write_json_file(file, dicts):
    """Write dict or dicts to a json file."""
    with open(file, 'w', encoding='utf-8', newline='') as fp:
        json.dump(dicts, fp, ensure_ascii=False)


if __name__ == '__main__':
    pass