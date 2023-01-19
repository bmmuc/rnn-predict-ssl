import os
import ipdb

def rename_file(dir, thread = False):
    i = 26549 + 1
    j = 0
    for file in os.listdir(f'{dir}'):
        if thread:
            #convert byte to kbytes
            # os.rename(f'{dir}/{file}', f'{dir}/{i}.jpg')
            if (os.path.getsize(f'{dir}/{file}')) / 1000 > 300:
                print(f'{dir}/{file}')
                # os.remove(f'{dir}/{file}')
                j += 1
        else:
            # if(file.count('-') == 2):
                # _, pos = file.split('-')
                # print('a')
            # else:
                # _, pos = file.split('-')
                # print(pos)
            
            # print(file)

            os.rename(f'{dir}/{file}', f'{dir}/positions-{i}.txt')
            # pos, _ = pos.split('.')
            i += 1
            # os.rename(f'{dir}/positions-{pos}.txt', f'{dir}/positions-{i}.txt')

        # i += 1
    print(f'{j} large files')
    print(f'{i} files renamed')

if __name__ == '__main__':
    rename_file('../all_data/data-3v3-v11', thread = False)