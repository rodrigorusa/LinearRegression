from reader.csv_reader import DiamondCsvReader


def main():
    diamond_list = DiamondCsvReader.read_csv('../diamonds-test.csv')

    DiamondCsvReader.get_data_frame('../diamonds-test.csv')

    for diamond in diamond_list:
        print(diamond, end='\n', flush=True)


if __name__ == '__main__':
    main()
