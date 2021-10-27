#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 16:59:51 2021

@author: gboleda
"""

# Code for figure adapted from
# https://matplotlib.org/stable/gallery/mplot3d/text3d.html

import matplotlib.pyplot as plt

words = ("courageous", "music", "heartbreak", "cub")
valences = (8.05, 7.67, 2.45, 6.71)
arousals = (5.5, 5.57, 5.65, 3.95)
dominances = (7.38, 6.5, 3.58, 4.24)

plt.close('all')
ax = plt.figure().add_subplot(projection='3d')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_zlim(0, 10)
ax.set_xlabel('x: valence')
ax.set_ylabel('y: arousal')
ax.set_zlabel('z: dominance')
ax.set_title('Words in the space of affective meaning')

print()
for word, x, y, z in zip(words, valences, arousals, dominances):
    ax.text(x, y, z, word, color='blue')
    print(word, x, y, z)

plt.show()
plt.close()
