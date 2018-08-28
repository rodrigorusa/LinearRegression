import csv

from model.diamond import Diamond



class DiamondCsvReader:
    @staticmethod
    def readCsv():
        diamondList = []

        with open('../diamonds.csv', mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)

            for row in csv_reader:
                diamond = Diamond(row[''], row['carat'], row['cut'], row['color'], row['clarity'], \
                                  row['depth'], row['table'], row['x'], row['y'], row['z'], row['price'])
                diamondList.append(diamond)


        return diamondList
