# Constant shift embedding for graph layout

Playing around with constant shift embedding method for graph layout

It doesn't look that bad!

![embedded graph](embedding.png)

TODO:

* Document
* Think more carefully about clipping strategies for infinite (and
  large) distances
* Improve efficiency:
    * Implicit symmetry in graph generation, Floyd's algorothm, and 
      embedding algorithm (centralisation step may be problematic?)
    * Can I reduce the number of calls to eigh from 2 to 1?
* Improve visualisation
