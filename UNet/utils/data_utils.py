import opendatasets as open
import zipfile
import shutil

def data_pull(data_url):
    open.download(data_url)

def unzip(input_path,export_path):
    with zipfile.ZipFile(input_path, 'r') as zip_ref:
        zip_ref.extractall(export_path)

def file_move(input_path,export_path):  
    shutil.move(input_path,export_path)