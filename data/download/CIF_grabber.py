import csv
from pymatgen.ext.matproj import MPRester
from pymatgen.io.cif import CifFile
from pymatgen.io.cif import CifWriter
import time


API_KEY = 'YOUR API KEY'
IDS = [[], []]


def getMaterialsInFile(csvfile):
    with MPRester(API_KEY) as m:
        for _id in csvfile:
            _id = _id[0]
            struct = m.get_structure_by_material_id(_id)
            file = CifWriter(struct, symprec=1.0)
            file.write_file(f'cifs/{_id}.cif')
            IDS[0].append(_id)
            IDS[1].append(1)
            time.sleep(1)


def main():
    file_names = ['need.csv']
    for file in file_names:
        with open(file, 'r', newline='') as csvfile:
            csvreader = csv.reader(csvfile)
            getMaterialsInFile(csvreader)


if __name__ == '__main__':
    main()
