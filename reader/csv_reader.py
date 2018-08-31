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
    def getDataFrame(fileName, power = [1, 1, 1, 1, 1, 1, 1, 1, 1]):
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
            dictionary['carat'].append(diamond.carat**power[0])
            dictionary['cut'].append(diamond.cut**power[1])
            dictionary['color'].append(diamond.color**power[2])
            dictionary['clarity'].append(diamond.clarity**power[3])
            dictionary['x'].append(diamond.x**power[4])
            dictionary['y'].append(diamond.y**power[5])
            dictionary['z'].append(diamond.z**power[6])
            dictionary['depth'].append(diamond.depth**power[7])
            dictionary['table'].append(diamond.table**power[8])
            dictionary['price'].append(diamond.price)

        return pd.DataFrame(dictionary, columns=columns)