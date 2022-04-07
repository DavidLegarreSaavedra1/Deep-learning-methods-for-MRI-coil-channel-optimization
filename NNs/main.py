import os
import pydicom as dicom


if __name__ == '__main__':

    data_path = os.path.join(os.getcwd(), *["data", "Kaggle"])

    print(data_path)