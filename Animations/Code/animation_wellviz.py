########################################
########### ANIMATION ############
dates = list(enumerate(df['production_date'].unique()))
num = 0
data = np.array(df[(df['pair_name'] == 'SA1_SA2') & (df['production_date'] == dates[date_num][1])]['dly_stm'])
n, bins = np.histogram(data, 100)

# get the corners of the rectangles for the histogram
left = np.array(bins[:-1])
right = np.array(bins[1:])
bottom = np.zeros(len(left))
top = bottom + n
nrects = len(left)

# Setting initial vertices and path codes
nverts = nrects * (1 + 3 + 1)
verts = np.zeros((nverts, 2))
codes = np.ones(nverts, int) * path.Path.LINETO
codes[0::5] = path.Path.MOVETO
codes[4::5] = path.Path.CLOSEPOLY
verts[0::5, 0] = left
verts[0::5, 1] = bottom
verts[1::5, 0] = left
verts[1::5, 1] = top
verts[2::5, 0] = right
verts[2::5, 1] = top
verts[3::5, 0] = right
verts[3::5, 1] = bottom

# patch setting
patch = None
# plt.hist(df[(df['pair_name'] == 'SA1_SA2')]['inj_tubing_pressure'], bins = 250)
# plt.hist(sorted(df[(df['pair_name'] == 'SA1_SA2')]['dly_stm']), bins = 250)
# Process animation
def animate(i):
    print("i = " + str(i))
    data = np.array(sorted(df[(df['pair_name'] == 'SA1_SA2') & (df['production_date'] == dates[i][1])]['dly_stm']))
    n, bins = np.histogram(data, 250)
    top = bottom + n
    verts[1::5, 1] = top
    verts[2::5, 1] = top
    return [patch, ]

###############################################################################
# And now we build the `Path` and `Patch` instances for the histogram using
# our vertices and codes. We add the patch to the `Axes` instance, and setup
# the `FuncAnimation` with our animate function.
fig, ax = plt.subplots()
barpath = path.Path(verts, codes)
patch = patches.PathPatch(
    barpath, facecolor='green', edgecolor='yellow', alpha=0.5)
ax.add_patch(patch)

ax.set_xlim(left[0], right[-1])
ax.set_ylim(bottom.min(), top.max())

ani = animation.FuncAnimation(fig, animate, int(0.2 * len(dates)), repeat=False, blit=True)

try:
    writer = animation.writers['ffmpeg']
except KeyError:
    writer = animation.writers['avconv']

writer = writer(fps=60)
ani.save('NewMovie.mp4', writer = writer)
