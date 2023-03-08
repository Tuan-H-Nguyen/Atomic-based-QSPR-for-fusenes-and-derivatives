import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FormatStrFormatter
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple


#plt font initialize
annotate = {'fontname':'Times New Roman','weight':'bold','size':20}
tick = {'fontname':'Times New Roman','size':13}
font = FontProperties()
font.set_weight('bold')
font_legend = font_manager.FontProperties(family = 'Times New Roman',size = 20)
  
def plot_histogram(
    dataset,
    save_path=None,
    label=None,
    label_loc=None,
    x_labels=True
    ):
    fig,ax = plt.subplots(nrows=1,ncols=1)
    ax.hist(
        dataset.loc[:]['Egap'],
        bins=[1.5,2,2.5,3,3.5,4,4.5,5]
        )
    ax.set_ylabel('Number of samples (samples)',**annotate)
    #
    ax.yaxis.set_major_locator(plt.MultipleLocator(10))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(5))
    #
    labely = ax.get_yticks().tolist()
    ax.yaxis.set_ticklabels(labely,**tick)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.f'))  
    #
    if x_labels:
        ax.set_xlabel('Band gap (eV)',**annotate)
        labelx = ax.get_xticks().tolist()
        ax.xaxis.set_ticklabels(labelx,**tick)
    else:
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.xaxis.set_minor_locator(plt.NullLocator())

    if label != None and label_loc != None:
        x,y = label_loc
        ax.text(x,y,label,**annotate)
    if save_path != None:
        fig.savefig(save_path,dpi=600,bbox_inches='tight')

class scatter_plot:
    def __init__(self,nrows = 1,ncols= 1,figsize = None):
        if figsize :
            assert isinstance(figsize,tuple)
            self.fig, self.ax = plt.subplots(
                nrows=nrows, ncols=ncols,
                figsize=figsize)
        else: 
            self.fig, self.ax = plt.subplots(nrows=nrows, ncols=ncols)
        self.lines = []
        self.scatters = []

    def add_plot(
        self,
        x,y,
        idx = 0,
        xlabel=None,
        ylabel=None,
        scatter = True,
        plot_line=None,
        weight=None,
        i=None,
        xticks_format=2,
        yticks_format=2,
        x_minor_tick=None,
        x_major_tick=None,
        y_minor_tick=None,
        y_major_tick=None,
        xlim=None, ylim=None,
        line_color = None,
        line_type = None,
        scatter_color=None,
        scatter_marker = None,
        scatter_size = 15,
        label =None,
        line_label = None,
        equal_aspect = False,
        tick_color = None
        ):
        ax = self.ax
        if scatter:
            scat = ax[idx].scatter(x,y,c = scatter_color, label = label,s=scatter_size, marker = scatter_marker)
            self.scatters.append(scat)
        if plot_line:
            if weight == None:
                line, = ax[idx].plot(x,y,linewidth=1.5, c=line_color, linestyle=line_type,label=line_label)
                self.lines.append(line)
            else:
                assert isinstance(weight,tuple)
                wb,w = weight
                if i == None:
                    i = np.linspace(min(x),max(x),1000)
                else:
                    assert isinstance(i,tuple)
                    i = np.linspace(i[0],i[1],100)
                line, = ax[idx].plot(i,wb+w*i,linewidth=1.5, c=line_color, linestyle=line_type, label = line_label)
                self.lines.append(line)
        if equal_aspect:
            self.fig.gca().set_aspect('equal',adjustable='box')
        #
        ax[idx].set_xlabel(xlabel,**annotate)
        if xlim:
            x,y = xlim
            ax[idx].set_xlim(x,y)
        if type(x_major_tick) is float or type(x_major_tick) is int:
            ax[idx].xaxis.set_major_locator(plt.MultipleLocator(x_major_tick))
        elif x_major_tick == 'null':
            ax[idx].xaxis.set_major_locator(plt.NullLocator())
        if x_minor_tick:
            ax[idx].xaxis.set_minor_locator(plt.MultipleLocator(x_minor_tick))
        try:
            labelx = ax[idx].get_xticks().tolist()
            ax[idx].xaxis.set_ticklabels(labelx,**tick)
            
            if xticks_format == -1:
                ax[idx].xaxis.set_major_formatter(plt.NullFormatter())
            else:
                xticks_format = '%.'+str(xticks_format)+'f'
                ax[idx].xaxis.set_major_formatter(FormatStrFormatter(xticks_format))
        except AttributeError:
            pass
            #
        ax[idx].set_ylabel(ylabel,**annotate)
        if ylim:
            x,y = ylim
            ax[idx].set_ylim(x,y)
        if type(y_major_tick) is float or type(y_major_tick) is int:
            ax[idx].yaxis.set_major_locator(plt.MultipleLocator(y_major_tick))
        elif y_major_tick == 'null':
            ax[idx].yaxis.set_major_locator(plt.NullLocator())
        if y_minor_tick:
            ax[idx].yaxis.set_minor_locator(plt.MultipleLocator(y_minor_tick))
        try:
            labely = ax[idx].get_yticks().tolist()
            ax[idx].yaxis.set_ticklabels(labely,**tick)
            
            if yticks_format == -1:
                ax[idx].yaxis.set_major_formatter(plt.NullFormatter())
            else:
                yticks_format = '%.'+str(yticks_format)+'f'
                ax[idx].yaxis.set_major_formatter(FormatStrFormatter(yticks_format))
        except AttributeError:
            pass
        #labely = ax[idx].get_yticks().tolist()
        #ax[idx].yaxis.set_ticklabels(labely,**tick)
        #yticks_format = '%.f' if yticks_format==0 else '%.'+str(yticks_format)+'f'
        #ax[idx].yaxis.set_major_formatter(FormatStrFormatter(yticks_format))
        if tick_color:
            ax[row_idx,col_idx].tick_params(axis='y',labelcolor=tick_color)
        
    def add_text(self,x,y,text,idx = 0,ha = "right",va = "bottom"):
        self.ax[idx].text(
            x,y,
            text,
            horizontalalignment=ha,
            verticalalignment=va,
            transform = self.ax[idx].transAxes,
            **annotate) 
    
    def add_text2(self,xr,yr,text,idx = 0):
        self.ax[idx].text(
            xr,yr,
            text,
            horizontalalignment='center',
            verticalalignment='center',
            transform = self.ax[idx].transAxes,
            **annotate) 


    def add_legend(self,loc = None,ncols=None):
        if loc == "None":
            self.ax.legend(prop = font_legend)
        elif loc == "above outside":
            self.ax.legend(
                prop = font_legend,
                loc="lower left",
                bbox_to_anchor=(0,1.02,1,0.2),
                mode="expand", borderaxespad=0,
                ncol = ncols
                )
        elif loc == 'left outside':
            self.ax.legend(
                prop = font_legend,
                loc = "center left",
                bbox_to_anchor=(1.04,0.5), borderaxespad=0)


    def save_fig(self,save_path,dpi=600):
        #self.ax.legend()
        self.fig.tight_layout()
        self.fig.savefig(save_path,dpi=dpi,bbox_inches="tight")

    def clear(self):
        self.fig.clf()
        del self.fig
