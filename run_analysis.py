from tools import *

def main():
#    copy_data()
#    return_dir()
    print "Doing some stuff..."
    seal_perc = ['p05','p25','p50','p75','p95']
    #seal_perc = ['p50']
    year_s = ['2050']
    copy_data()
    save_max_comp('RCP45NA',seal_perc,year_s)
    print "DONE!"

if __name__ == '__main__':
    main()
