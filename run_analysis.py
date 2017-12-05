from tools import *





def main(plot_type,exceedance_value=None):
#    copy_data()
#    return_dir()
    from numpy import sum
    import matplotlib.pyplot as plt
    print
    print "\t\033[1m Analyzing model results\033[0m"
    seal_perc = ['p05','p25','p50','p75','p95']

    #exceedance_value = 0.55
#    plot_type = 'flood'
    if plot_type == 'exceedance':
        print "      Exceedence probability for {t1}".format(t1=exceedance_value)
        eq1,data1 = calculate_exceedence(exceedance_value, 'RCP85WA', seal_perc,'2100')
        plt.plot(eq1,data1, lw=1)
        eq2,data2 = calculate_exceedence(exceedance_value, 'RCP85WA', seal_perc,'2050')
        plt.plot(eq2,data2, lw=1)
        eq3,data3 = calculate_exceedence(exceedance_value, 'RCP45NA', seal_perc,'2000')
        plt.plot(eq3,data3,lw=1)
        plt.ylim(-0.1,1.1)
        plt.xlim(8.0,9.4)
        plt.xlabel("Magnitude")
        plt.ylabel("Exceedance Probability ({t1} m)".format(t1=exceedance_value))
        plt.savefig('test.png',dpi=501, bbox_inches="tight")
    if plot_type == 'flood':
        print "\t Flood-level probability"
        x1,y1 = calculate_flood_probability('RCP45NA', seal_perc, '2000')
        print "  Integral of y1 = {t1}".format(t1=sum(y1))
        plt.plot(x1,y1,'b-',lw=1,label = '2000')

        x2,y2 = calculate_flood_probability('RCP45NA', seal_perc, '2050')
        print "  Integral of y2 = {t1}".format(t1=sum(y2))
        plt.plot(x2,y2,'k-',lw=1,label = '2050')
        #
        x3,y3 = calculate_flood_probability('RCP45NA', seal_perc, '2100')
        print "  Integral of y3 = {t1}".format(t1=sum(y3))
        plt.plot(x3,y3,'r-',lw=1,label = '2100')
        plt.title('RCP 4.5')
        plt.xlabel('Flood Levels [m]')
        plt.legend()
        plt.ylabel('PDF')
    plt.savefig('test.png',dpi=501, bbox_inches="tight")
    print "\t\tDONE!"

if __name__ == '__main__':
    main('exceedance',0.8)
