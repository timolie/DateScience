import csv

def write_missing_values_csv(missing_values, file):
    m_values = list(zip(missing_values.index,missing_values))

    with open(file, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(['Feature', 'Missing Values in %'])
        for row in m_values:
            writer.writerow(row)