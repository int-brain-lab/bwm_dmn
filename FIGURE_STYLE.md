# Figure style preferences

Use these defaults for figures prepared for Mic:

- Show at most three major ticks on each visible x- and y-axis.
- Hide top and right axes spines.
- Use clear bold lowercase panel labels (`a`, `b`, `c`, ...).
- Keep explanatory prose in the caption rather than as panel titles.
- Follow Nature primary-research artwork defaults: prepare at final print width
  (89 mm single column or 183 mm double column), use Arial/Helvetica sans serif,
  5--7 pt plot text, 8 pt bold upright panel letters, approximately 0.5 pt axes,
  and editable vector PDF output.

For Matplotlib, apply `MaxNLocator(nbins=2)` to visible axes (at most three tick
locations, including endpoints) unless fixed categorical ticks are required.
