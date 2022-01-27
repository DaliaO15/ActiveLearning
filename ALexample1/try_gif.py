from IPython import display
from IPython.display import Image

# Display GIF in Jupyter, CoLab, IPython
with open("rs1it5.gif", 'rb') as f:
    display.Image(data=f.read(), format='png')