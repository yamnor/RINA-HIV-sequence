from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

import numpy as np
from numpy import where
import pandas as pd

import streamlit as st

from matplotlib import pyplot as plt
from matplotlib import collections

import io

nres = 200

def mkfig(w, h):
  figsize = [w, h]
  subplot = {
    'left':   0.10,
    'right':  0.10,
    'bottom': 0.10,
    'top':    0.10,
    'wspace': 1.50,
    'hspace': 2.00,
    'grid': True,
  }
  with plt.style.context('matplotlibrc'):
    plt.rcParams["figure.figsize"]        = figsize
    plt.rcParams["figure.subplot.left"]   = subplot['left'] / figsize[0]
    plt.rcParams["figure.subplot.right"]  = 1.00 - subplot['right'] / figsize[0]
    plt.rcParams["figure.subplot.bottom"] = subplot['bottom'] / figsize[1]
    plt.rcParams["figure.subplot.top"]    = 1.00 - subplot['top'] / figsize[1]
    plt.rcParams["figure.subplot.wspace"] = subplot['wspace'] / figsize[0]
    plt.rcParams["figure.subplot.hspace"] = subplot['hspace'] / figsize[1]
    plt.rcParams["axes.grid"]             = subplot['grid']
    fig, ax = plt.subplots()
  return fig, ax

def main():

  key = {'hb': 0, 'vdw': 1, 'ss': 2, 'ion': 3, 'pp': 4, 'pc': 5, 'iac': 6}

  st.title("RIN Analyzer for HIV-1 protease")

  xy = np.loadtxt('data/node.txt')
  
  theta = 175 * 3.14 / 180
  xy = np.array([[np.cos(theta) * x[0] - np.sin(theta) * x[1], np.sin(theta) * x[0] + np.cos(theta) * x[1]] for x in xy])
  node = pd.DataFrame(xy, columns=['x', 'y'])
  node['chain'] = ['A' for n in range(99)] + ['B' for n in range(99)] + ['C']
  node['resid'] = [n+1 for n in range(99)] + [n+1 for n in range(99)] + [1]

  seq_file = st.file_uploader("Choose a RIN sequence file")

  if seq_file is not None:

    stringio = io.StringIO(seq_file.getvalue().decode("utf-8"))
    sequence = [line.split()[1:] for line in stringio.readlines()]
    nsamples = len(sequence)

    rin_items = st.multiselect(
      'Select interactions',
      ['hb', 'vdw', 'ss', 'ion', 'pp', 'pc', 'iac'],
      ['hb', 'vdw'])

    nitems = len(rin_items)

    rinseq = []
    rinmat = []
    rinsum = np.zeros((nitems, nres, nres))
    for t in range(nsamples):
      tmp = np.zeros((nitems, nres, nres))
      for n in range(len(sequence[t]) // 3):
        xt = sequence[t]
        rx = xt[n * 3]
        ix = int(xt[n * 3 + 1])
        jx = int(xt[n * 3 + 2])
        for k in range(nitems):
          if rin_items[k] == rx:
            tmp[k][ix][jx] = 1
            rinsum[k][ix][jx] += 1.0 / nsamples
      rinmat.append(tmp)
      rinseq.append(np.ravel(tmp))
    xy = TSNE(n_components = 2, init = 'pca', random_state = 0).fit_transform(np.array(rinseq))

    rinmat = np.array(rinmat)

    nclusters = st.number_input('Number of Clusters', 2, 6, 3, 1)

    xy_class = KMeans(n_clusters = nclusters).fit_predict(rinseq)

    fig, ax = mkfig(6, 4)
    for i in range(nclusters):
      ax.scatter(*xy[where(xy_class == i)].T)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    for axis in ['top', 'left', 'bottom', 'right']:
      ax.spines[axis].set_linewidth(0.1)
    st.pyplot(fig)

    rinsum_ref = np.zeros((nitems, nres, nres))

    for c in range(nclusters):
      rinmat_c = rinmat[where(xy_class == c)]
      col = st.columns(nitems)
      for item in range(nitems):
        lines = []
        fracs = []
        color = []
        for i in range(nres):
          for j in range(i+1, nres):
            sum = 0.0
            for t in range(len(rinmat_c)):
              sum += rinmat_c[t][item][i][j]
            sum /= len(rinmat_c)
            if c == 0:
              rinsum_ref[item][i][j] = sum
            if sum > 0:
              lines.append([(node.at[i-1, 'x'], node.at[i-1, 'y']), (node.at[j-1, 'x'], node.at[j-1, 'y'])])
              if c == 0:
                val = sum
              else:
                val = sum - rinsum_ref[item][i][j]
              fracs.append(abs(val))
              if val > 0.0:
                color.append('b')
              else:
                color.append('r')

        lc = collections.LineCollection(lines, linewidth = fracs, colors = color, alpha = 0.5)

        with col[item]:
          if c == 0:
            st.write(f'group: {c}, item: {rin_items[item]} (REFERENCE)')
          else:
            st.write(f'group: {c}, item: {rin_items[item]}')
          fig, ax = mkfig(6, 4)
          for chain in ['A', 'B', 'C']:
            ax.scatter(node[node['chain'] == chain].x, node[node['chain'] == chain].y, s = 25, alpha = 0.5, label = chain)
          ax.add_collection(lc)
          ax.xaxis.set_visible(False)
          ax.yaxis.set_visible(False)
          for axis in ['top', 'left', 'bottom', 'right']:
            ax.spines[axis].set_linewidth(0.1)
          st.pyplot(fig)

plt.legend()

if __name__ == "__main__":
    main()
