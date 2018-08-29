import csv
import pandas as pd

from model.diamond import Diamond

class DiamondCsvReader:
    @staticmethod
    def readCsv(fileName):
        diamondList = []

        with open(fileName, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)

            for row in csv_reader:
                diamond = Diamond(row['carat'], row['cut'], row['color'], row['clarity'], \
                                  row['depth'], row['table'], row['x'], row['y'], row['z'], row['price'])
                diamondList.append(diamond)

        return diamondList

    @staticmethod
    def getDataFrame(fileName):
        diamondList = DiamondCsvReader.readCsv(fileName)

        dictionary = {
            'carat': [],
            'cut': [],
            'color': [],
            'clarity': [],
            'x': [],
            'y': [],
            'z': [],
            'depth': [],
            'table': [],
            'price': []
        }

        columns = ['carat', 'cut', 'color', 'clarity', 'x', 'y', 'z', 'depth', 'table', 'price']

        for diamond in diamondList:
            dictionary['carat'].append(diamond.carat)
            dictionary['cut'].append(diamond.cut)
            dictionary['color'].append(diamond.color)
            dictionary['clarity'].append(diamond.clarity)
            dictionary['x'].append(diamond.x)
            dictionary['y'].append(diamond.y)
            dictionary['z'].append(diamond.z)
            dictionary['depth'].append(diamond.depth)
            dictionary['table'].append(diamond.table)
            dictionary['price'].append(diamond.price)

        return pd.DataFrame(dictionary, columns=columns)