import argparse
import Levenshtein
import csv


def calculate_edit_distance(file1, file2, output_file):
    with open(file1, 'r') as f1, open(file2, 'r') as f2, open(output_file, 'w', newline='') as out_file:
        domains1 = f1.read().splitlines()
        domains2 = f2.read().splitlines()

        writer = csv.writer(out_file, delimiter='\t')
        writer.writerow(["Domain1", "Domain2", "Levenshtein Distance"])

        for domain1 in domains1:
            for domain2 in domains2:
                distance = Levenshtein.distance(domain1, domain2)
                writer.writerow([domain1, domain2, distance])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Levenshtein distance between domains in two files and save the results in TSV format.")
    parser.add_argument("file1", help="Path to the first file containing domain names.")
    parser.add_argument("file2", help="Path to the second file containing domain names.")
    parser.add_argument("output_file", help="Path to the output TSV file.")
    args = parser.parse_args()

    calculate_edit_distance(args.file1, args.file2, args.output_file)
