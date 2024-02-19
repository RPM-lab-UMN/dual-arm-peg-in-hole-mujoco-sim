for SHAPE in 'Arrow' 'Circle' 'Cross' 'Diamond' 'Hexagon' 'Key' 'Line' 'Pentagon' 'U'
do
    python convex_hull_decomp.py -i ../${SHAPE}_cube_bottle_preprocessed_convex.obj -o ${SHAPE}_cube_bottle_decomp.obj -t 0.04 -pr 100
    python convex_hull_decomp.py -i ../${SHAPE}_cube_cap_preprocessed_convex.obj -o ${SHAPE}_cube_cap_decomp.obj -t 0.04 -pr 100
done
