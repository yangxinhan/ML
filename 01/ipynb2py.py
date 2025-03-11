import os,sys,json

def convert_nb(input_file_name):
    input_file = open(input_file_name, 'r', encoding='utf8') 
    out_file_name = input_file_name.replace('.ipynb', '.py').replace('\\', '/').split('/')[-1]
    out_file = open(out_file_name, 'w', encoding='utf8') 
    input_json = json.load(input_file)
    for i,cell in enumerate(input_json["cells"]):
        # out_file.write("#cell "+str(i)+"\n")
        for line in cell["source"]:
            if cell["cell_type"] != 'code' and not line.startswith('#'):
                out_file.write('### ')
            elif cell["cell_type"] == 'code' and line.startswith('!'):
                out_file.write('# ')

            out_file.write(line)
        if cell["cell_type"] == 'code':
            out_file.write('\n\n')
        else:
            out_file.write('\n')
        
    out_file.close()    
    
if __name__ == '__main__' :
    input_file_name = sys.argv[1] 
    if os.path.isdir(input_file_name):
        for file_name in os.listdir(input_file_name):
            if not file_name.endswith('ipynb'): continue
            convert_nb(os.path.join(input_file_name, file_name))
    else:    
        convert_nb(input_file_name)    