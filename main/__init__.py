from reader.csv_reader import DiamondCsvReader

def main():
    diamondList = DiamondCsvReader.readCsv()

    for diamond in diamondList:
        print(diamond, end='\n', flush=True)


if __name__ == '__main__':
    main()