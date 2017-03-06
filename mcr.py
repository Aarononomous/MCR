from math import *
import shapely as sh
from shapely.geometry import *
from shapely.affinity import *
from shapely.prepared import prep
import graphviz as gv
import random as rand
import matplotlib.pyplot as plt
import re
import networkx as nx
from itertools import combinations


class MCR:

    """
    Minimum Cover Removal. Contains a number of helper methods for
    investigating the MCR problem as described in Erickson and LaValle 2013
    (EL13) and Hauser 2013 (H13)

    EL13: https://www.semanticscholar.org/paper/A-Simple-but-NP-Hard-Motion-Planning-Problem-Erickson-LaValle/0a9a3a6249eea0cf31646a1c97c822c0213381b7
    H13: https://pdfs.semanticscholar.org/153e/a4fb187bd0dbda27a51979ff8f09c478bf59.pdf
    """

    # Styling defaults for drawing obstacles and graphs. These can be
    # overriden by setting myMCR.nx_opts, e.g., to your preferred style.

    # Obstacles are drawn in light blue, 15% opacity to show overlapping
    shape_opts = {'alpha': 0.15,
                  'edgecolor': '#336699',
                  'facecolor': '#77ccff'}

    # The start and goal points are red
    point_opts = {'color': '#FF2233'}

    # Nodes and edges in the graph are the same red, and small and narrow
    # enough not to distract
    nx_opts = {'node_size': 33,
               'node_color': '#FF2233',
               'width': 0.6667,
               'edge_color': '#FF2233'}

    # Axes options
    tick_params = {'axis': 'both',  # changes apply to both axes
                   'which': 'both',  # major and minor ticks are affected
                   'top': 'off',  # ticks along the top edge are off
                   'bottom': 'off',  # ditto
                   'left': 'off',  # ditto
                   'right': 'off',  # ditto
                   'labelbottom': 'off',  # no labels on the bottom
                   'labelleft': 'off'}  # ditto

    def __init__(self, svg=None):
        """
        Create an empty square to add shapes to.
        """

        # Public members
        self.obstacles = []  # a list of the Polygon obstacles
        self.overlapped_obstacles = []  # all overlappings; also Polygons
        self.field = Polygon([(0,0), (0,1), (1,1), (1,0)])
        self.graph = nx.Graph()
        self.start = Point(0.01, 0.01)
        self.goal = Point(0.99, 0.99)

        # Private members
        self._obs_count = 1  # For labeling shapes uniquely
        self._current = False  # Do overlaps need to be recalculated?
        self._current_graph = False  # Does graph need to be recalculated?

        if svg:  # initialize from svg file
            try:
                obstacles = MCR.__parse_SVG(svg)
                for o in obstacles:
                    self.add_obstacle(o)
            except:
                print('Couldn\'t load shapes from {}'.format(svg))
                raise ValueError

    def add_obstacle(self, shape):
        """
        Adds new labeled obstacles to the field.
        """
        if not hasattr(shape, 'cover'):
            # The "cover" of a node in the constructed graph is a set
            # containing the label of every obstacle the node is within. We
            # simply add this as an attribute to the shape itself.
            shape.cover = set([self._obs_count])
            self._obs_count += 1

        self.obstacles.append(shape)
        self._current = False
        self._current_graph = False

    def remove_obstacle(self, label):
        """
        Removes the obstacle labeled 'label'.
        Removes the labeled vertices from the graph and contracts those edges
        """
        # N.B.: numbering starts at 1, and we assume that all obstacles are
        # labeled with a number
        self.obstacles = self.obstacles[:label-1] + self.obstacles[label:]
        self._current = False
        self._current_graph = False

    def construct_overlaps(self):
        """
        Refreshes the overlapped_obstacles list. Each obstacle is added one at
        a time to overlapped_obstacles. If there are any overlaps with any
        already-present obstacles, their intersection is added. Each new shape
        is labelled with the union of the covers from its constituent
        obstacles.
        """
        overlapped_obstacles = [self.obstacles[0]]
        union = self.obstacles[0]  # the union of all obstacles

        for obstacle in self.obstacles[1:]:
            # prepare obstacle and determine any overlaps
            # remember the cover -- geometric operations create new polygons
            # without them
            cover = obstacle.cover
            obstacle &= self.field  # trim to self.field
            prepped = prep(obstacle)  # prepping speeds up the test
            overlapped = [o for o in overlapped_obstacles
                          if prepped.intersects(o)]

            for o_o in overlapped:
                # first remove the overlapped obstacle, then re-add, piece by
                # overlapped piece
                overlapped_obstacles.remove(o_o)
                unoverlapped = o_o - obstacle
                overlapped = o_o & obstacle

                # re-add the unoverlapped parts, but only polygons, not lines
                if type(unoverlapped) is Polygon:
                    unoverlapped.cover = o_o.cover
                    overlapped_obstacles.append(unoverlapped)
                elif type(unoverlapped) in [MultiPolygon, GeometryCollection]:
                    for u in unoverlapped:
                        if type(u) is Polygon:
                            u.cover = o_o.cover
                            overlapped_obstacles.append(u)

                # add the overlapped section(s), labeling them as covered by
                # both obstacles
                if type(overlapped) is Polygon:
                    overlapped.cover = o_o.cover | cover
                    overlapped_obstacles.append(overlapped)
                elif type(overlapped) in [MultiPolygon, GeometryCollection]:
                    for o in overlapped:
                        if type(o) is Polygon:
                            o.cover = o_o.cover | cover
                            overlapped_obstacles.append(o)

            # any part of the obstacle that's not overlapped another can be
            # added now
            new_obstacles = obstacle - union

            if type(new_obstacles) == Polygon:
                new_obstacles.cover = cover
                overlapped_obstacles.append(new_obstacles)
            elif type(new_obstacles) in [MultiPolygon, GeometryCollection]:
                for n in new_obstacles:
                    if type(n) == Polygon:
                        n.cover = cover
                        overlapped_obstacles.append(n)
            union |= obstacle  # update the union to contain all obstacles

        self.overlapped_obstacles = overlapped_obstacles
        self._current = True

    def plot_obstacles(self, labels=False):
        """
        Outputs the original obstacles and start and goal points, without
        highlighting intersections.
        """
        MCR.plot_shapes(self.obstacles, labels)
        MCR.plot_points([self.start, self.goal])

    def plot_overlapped_obstacles(self, labels=False):
        """
        Outputs the overlapped obstacles and start and goal points. This
        generally looks much "flatter" than plot_obstacles() because of the
        lack of overlapping.
        """
        if not self._current:
            self.construct_overlaps()

        MCR.plot_shapes(self.overlapped_obstacles, labels)
        MCR.plot_points([self.start, self.goal])

    def intersections_of(self, label):
        """
        Returns a list of any other obstacles which intersect this one
        """
        prepped = prep(self.obstacles[label - 1])
        return [o_o for o_o in self.obstacles if prepped.intersects(o_o)]

    def create_graph(self):
        '''
        Create the intersection graph. This should be done anytime an object is
        added or removed and whenever the start/goal locations change.
        '''
        if not self._current:
            self.construct_overlaps()

        countries = self.overlapped_obstacles[:]  # create a copy

        # label and append any remaining whitespace to polygons in countries
        field = Polygon(self.field)  # create a copy
        for o in self.obstacles:
            field -= o

        # the cover of the field is ∅
        if type(field) == Polygon:
            field.cover = set()
            countries.append(field)
        else:
            for f in field:
                f.cover = set()
                countries.append(f)

        # refresh current graph
        G = self.graph
        G.clear()

        # Add each country as a node in the graph. Neighboring countries are
        # connected with edges
        for country in countries:
            # Polygons aren't hashable -- they're mutable -- but their well-
            # known texts are
            wkt = country.wkt
            G.add_node(wkt)
            pt = country.representative_point()
            G.node[wkt]['pos'] = (pt.x, pt.y)
            G.node[wkt]['cover'] = country.cover
            G.node[wkt]['label'] = ','.join(map(str, country.cover))

            # Do these overlap in a line(s) (1-D overlap)? Use the DE-9IM
            # relationship: http://giswiki.hsr.ch/images/3/3d/9dem_springer.pdf
            adjacencies = [x for x in countries if country is not x
                           and country.relate(x)[4] == '1']
            for adj in adjacencies:
                G.add_edge(wkt, adj.wkt)

        # Add start and goal
        G.add_node(self.start.wkt)
        G.node[self.start.wkt]['pos'] = (self.start.x, self.start.y)
        G.node[self.start.wkt]['cover'] = set()
        G.node[self.start.wkt]['label'] = 'start'

        G.add_node(self.goal.wkt)
        G.node[self.goal.wkt]['pos'] = (self.goal.x, self.goal.y)
        G.node[self.goal.wkt]['cover'] = set()
        G.node[self.goal.wkt]['label'] = 'goal'

        for country in countries:
            if country.contains(self.start):
                G.add_edge(self.start.wkt, country.wkt)
            if country.contains(self.goal):
                G.add_edge(self.goal.wkt, country.wkt)

        self._current_graph = True

    def plot_graph(self, labels=False):
        """
        Plots the field. Note that none of these "plot_..." methods call
        plt.show(). That's done only after everything's been added to plt.
        """
        if not self._current:
            self.construct_overlaps()

        if not self._current_graph:
            self.create_graph()

        pos = nx.get_node_attributes(self.graph, 'pos')
        nx.draw_networkx(self.graph, pos, with_labels=False, **MCR.nx_opts)

        if labels:  # labels are offset below and right of nodes
            l_p = {n: (x + 0.025, y - 0.025) for n, (x, y) in pos.items()}
            nx.draw_networkx_labels(self.graph, l_p,
                                    labels=nx.get_node_attributes(
                                        self.graph, 'label'),
                                    **MCR.nx_opts)

    def show(self, obstacles=True, graph=True, labels=False):
        """
        The default display method. If you want to show a more complicated
        situation, build your output piece by piece, then call plt.show().
        """
        if obstacles:
            self.plot_obstacles(labels=(not graph))

        if graph:
            self.plot_graph(labels=labels)

        MCR.setup_axes()
        plt.show()

    # Static methods on MCR

    @staticmethod
    def setup_axes(**tick_params):
        """
        A helper method for displaying things nicely.
        Sets up a plotting scheme with a black outline, no labels, and no ticks
        on the axes.
        """
        params = tick_params if len(tick_params) else MCR.tick_params
        plt.tick_params(**params)
        plt.axis([0, 1, 0, 1])  # set the axes to [0,1]

    @staticmethod
    def plot_points(points):
        """
        Plots shapely Points
        """
        xs = [pt.x for pt in points]
        ys = [pt.y for pt in points]
        plt.scatter(xs, ys, **MCR.point_opts)

    @staticmethod
    def plot_linestrings(linestrings):
        """
        Plots shapely LineStrings. A LineString is simply a list of points.
        """
        for linestring in linestrings:
            for line in zip(linestring.coords, linestring.coords[1:]):
                plt.plot(*zip(*line))  # I know...

    @staticmethod
    def plot_shapes(shapes, labels=False):
        """
        Plots shapely Polygons (the exterior only).
        """
        for s in shapes:
            # set the styling of the object from the defaults and any
            # overridden attributes
            opts = MCR.shape_opts.copy()
            if hasattr(s, 'facecolor'):
                opts['facecolor'] = s.facecolor
            if hasattr(s, 'edgecolor'):
                opts['edgecolor'] = s.edgecolor

            poly = plt.Polygon(s.exterior.coords, **opts)
            plt.gca().add_patch(poly)

            # label shapes
            if labels and hasattr(s, 'cover'):
                r_p = s.representative_point()
                plt.text(r_p.x, r_p.y, ','.join(map(str, s.cover)),
                         horizontalalignment='center',
                         verticalalignment='center')

    @staticmethod
    def __parse_SVG(svg_file):
        """
        Reads in and parses an SVG file. Returns a list of Polygons resized to
        fit within a [0,1] - [0,1] field.
        Note that there are tons of SVG shapes this can't handle -- also, I'm
        parsing with regex, which can't be good.
        """
        try:
            f = open(svg_file)
            svg = f.read()
            f.close()
        except FileNotFoundError:
            print('File {} not found'.format(svg_file))
            return

        shapes = []  # parsed shapes are accumulated in this list

        # Viewbox
        try:
            viewbox_ = re.findall('viewBox="(.*?)"', svg)[0]
            viewbox = [float(x) for x in viewbox_.split()]
            vb_w = viewbox[2] - viewbox[0]
            vb_h = viewbox[3] - viewbox[1]
        except:
            print('Can\'t find a viewbox!')
            return

        # Rectangles
        rects = re.findall('<rect.*?\/>', svg)
        for r in rects:
            x_ = re.findall('x="([-\d.]+)"', r)
            x = float(x_[0]) if x_ else 0.0

            y_ = re.findall('y="([-\d.]+)"', r)
            y = float(y_[0]) if y_ else 0.0

            h_ = re.findall('height="([\d.]+)"', r)
            h = float(h_[0])

            w_ = re.findall('width="([\d.]+)"', r)
            w = float(w_[0])

            rect = Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])

            # Transforms on rectangles
            transform = re.findall('transform="(.*?)"', r)
            if transform:
                t_list = re.findall('\w+\(.+?\)', transform[0])

                # reverse t_list to compose transforms
                for t in reversed(t_list):
                    if t.startswith('matrix'):
                        vals = re.findall('\((.+)\)', t)[0]
                        a, d, b, e, xoff, yoff = [
                            float(x) for x in re.split(',?\s+', vals)]
                        rect = affine_transform(
                            rect, [a, b, d, e, xoff, yoff])

                    elif t.startswith('translate'):
                        vals = re.findall('\((.+)\)', t)[0]
                        x, y = (float(x) for x in re.split('\s+,?\s*', vals))
                        rect = translate(rect, x, y)

                    elif t.startswith('scale'):
                        vals = re.findall('\((.+)\)', t)[0]
                        scales = [float(x) for x in re.split('\s+,?\s*', vals)]
                        if len(scales) == 1:  # expand 1-arg shorthand notation
                            scales[1] = scales[0]
                        rect = scale(rect, *scales, origin=(0,0))

                    elif t.startswith('rotate'):
                        rot = re.findall('\((.+)\)', t)[0]
                        rect = rotate(rect, float(rot), origin=(0,0))

                    elif t.startswith('skewX'):
                        skew_x = re.findall('\((.+)\)', t)[0]
                        rect = skew(rect, xs=float(skew_x), origin=(0,0))

                    elif t.startswith('skewY'):
                        skew_y = re.findall('\((.+?)\)', t)[0]
                        rect = skew(rect, ys=float(skew_y), origin=(0,0))

            shapes.append(rect)

        # Circles
        circles = re.findall('<circle.*?\/>', svg)
        for c in circles:
            cx_ = re.findall('cx="([-\d.]+)"', c)
            cx = float(cx_[0]) if cx_ else 0.0

            cy_ = re.findall('cy="([-\d.]+)"', c)
            cy = float(cy_[0]) if cy_ else 0.0

            r_ = re.findall('r="([-\d.]+)"', c)
            r = float(r_[0]) if r_ else 0.0

            circle = Point(cx, cy).buffer(r)  # buffer is a shapely idiom
            shapes.append(circle)

        # Polygons
        polygons = re.findall('<polygon.*?\/>', svg)
        for p in polygons:
            point_list = re.findall('points="(.*?)"', p)

            # ignore the last point: it repeats the first
            points = ([float(x) for x in point_list[0].split()])[:-2]
            shapes.append(Polygon(list(zip(points[::2], points[1::2]))))

        # rescale all parsed shapes to between [0,0] and [1,1]
        # additionally, the SVG viewbox origin is at the top left, i.e., upside
        # down, so scale it by -1 in the x direction and translate it back up
        # to the regular Cartesian system
        scaled_shapes = []
        for shape in shapes:
            shape = translate(scale(shape,
                                    xfact=1 / vb_w,
                                    yfact=-1 / vb_h,
                                    origin=(0,0)),
                              0, 1)
            scaled_shapes.append(shape)

        return scaled_shapes


def random_MCR(obstacles=10, scale_factor=0.25):
    """
    Creates an MCR with random polygonal obstacles.
    Shapes are gotten from the random_shape function
    """
    mcr = MCR()
    pts = poisson_2d(obstacles)  # a list of random (x,y) points
    random_obstacles = []

    for _ in range(obstacles):
        shape = scale(random_shape(), scale_factor, scale_factor)
        x_min, y_min, _, _ = shape.bounds
        t_x, t_y = pts.pop()
        random_obstacles.append(translate(shape, t_x - x_min, t_y - y_min))

    # rescale again, to fit all shapes into the unit field
    bounds = [o.bounds for o in random_obstacles]
    max_x = max([b[2] for b in bounds])
    max_y = max([b[3] for b in bounds])

    for obstacle in random_obstacles:
        mcr.add_obstacle(scale(obstacle, 1 / max_x, 1 / max_y, origin=(0,0)))

    return mcr


def random_shape(max_sides=7):
    """
    Creates a random convex polygon with between 3 and ~max_sides sides.
    """
    return rotate(approx_ngon(rand.randrange(3, max_sides)),
                  rand.uniform(0, 360),
                  origin='centroid')


def on_left_hand(point, ray):
    """
    Is point on or on the left-hand side of the directed ray ray_0 -> ray_1?
    """
    px, py = point
    x1, y1 = ray[0]
    x2, y2 = ray[1]

    # special case for vertical rays (no slope)
    if (x1 == x2):
        return (px < x1) == (y1 < y2)

    # find the slope and y-intersect, then see if point is above or below the
    # ray
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    dist = py - (m * px + b)

    # if it's above and the ray is rightward, or below and the ray is leftward,
    # it's on the left-hand side
    return (dist >= 0) == (x1 < x2)


def approx_ngon(n, variance=0.25):  # TODO: magic number
    """
    Creates a random polygon with approximately n sides of approximately
    length 1. It's guaranteed that the polygon will be convex. No other
    guarantees, though.
    """
    ngon = [(0,0), (1,0)]  # the polygon
    Σ_α = 0  # the total angle

    # while the newest point is on the left-hand side of ngon[0] - ngon[1] and
    # while ngon[0] is on the left-hand side of the most recent edge and
    # while \Sum a < 2π, continue to add edges
    while True:
        Σ_α += rand.gauss(2*pi / n, variance / 2)  # TODO: magic number
        l = rand.gauss(1, variance)
        last_x, last_y = ngon[-1]
        pt = (last_x + cos(Σ_α)*l, last_y + sin(Σ_α)*l)

        if Σ_α <= 2*pi and on_left_hand(pt, (ngon[:2]))\
                and on_left_hand(ngon[0], (ngon[-1], pt)):
            ngon.append(pt)
        else:
            return Polygon(ngon)


def poisson_2d(k):
    """
    Returns an array of k (x, y) tuples evenly spaced across [0,1] x [0,1]
    Note that there are tweaks to the distribution that make it unreliable
    as an _actual_ Poisson distribution.
    """
    xs = [0]
    ys = [0]

    for _ in range(k-1):
        xs.append((xs[-1] + rand.expovariate(k)))
        ys.append((ys[-1] + rand.expovariate(k)))

    xs = [x / max(xs) for x in xs]
    ys = [y / max(ys) for y in ys]

    rand.shuffle(ys)
    return list(zip(xs, ys))
