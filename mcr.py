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
    investigating the MCR problem as described in Erickson and LaValle 2013 and
    Hauser 2012
    """

    # We display obstacles by default as light gray with visible overlapping
    # The start and goal points are red
    shape_opts = {
        'alpha': 0.15, 'edgecolor': '#336699', 'facecolor': '#77ccff'}
    point_opts = {'color': 'red'}
    nx_opts = {'node_size': 33, 'node_color': '#FF2233',
               'width': 0.6667, 'edge_color': '#FF2233'}


    def __init__(self, svg=None):
        """
        Create an empty square to add shapes to.
        """
        self.obstacles = []
        self.overlapped_obstacles = []
        self.graph = nx.Graph()
        self.graph_labels = {}  # node labels
        self.graph_pos = {}  # node positions
        self.start = Point(0.15, 0.05)
        self.goal = Point(0.95, 0.95)
        self._label = 1  # Shapes are labeled either explicitly or using this
        self._current = True  # False if overlaps need to be recalculated

        self.field = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])

        if svg:  # initialize from svg file
            try:
                obstacles = MCR.__parse_SVG(svg)
                for o in obstacles:
                    self.add_obstacle(o)
            except:
                print('Couldn\'t load all shapes in ', svg)
                raise ValueError


    def add_obstacle(self, shape):
        """
        Adds new labeled obstacles to the field.
        """
        if not hasattr(shape, 'label'):
            shape.label = str(self._label)
            self._label += 1

        self.obstacles.append(shape)
        self._current = False


    def remove_obstacle(self, label):
        """
        Removes the obstacle labeled 'label'.
        Removes the labeled vertices from the graph and contracts those edges
        """
        # N.B.: numbering starts at 1
        self.obstacles = self.obstacles[:label-1] + self.obstacles[label:]
        self._current = False


    def construct_overlaps(self):
        """
        Refreshes the overlapped_obstacles list
        """
        overlapped_obstacles = []

        # for every shape that is added, it may intersect with any other
        for shape in self.obstacles:
            label = shape.label
            # trim to field bounds
            shape = shape & self.field
            prepped = prep(shape)  # speed up testing
            overlaps = [
                o_o for o_o in overlapped_obstacles if prepped.intersects(o_o)]

            for o_o in overlaps:
                # first remove, then re-add piece by piece
                overlapped_obstacles.remove(o_o)
                unoverlapped = o_o - shape
                overlapped = o_o & shape

                # we don't care about Points, Lines, etc.
                if unoverlapped.type == 'Polygon':
                    unoverlapped.label = o_o.label
                    overlapped_obstacles.append(unoverlapped)
                elif unoverlapped.type in ['MultiPolygon', 'GeometryCollection']:
                    for p in unoverlapped:
                        if p.type == 'Polygon':
                            p.label = o_o.label
                            overlapped_obstacles.append(p)

                # otherwise the new piece is wholly within unoverlapped
                if overlapped.type == 'Polygon':
                    overlapped.label = o_o.label + ',' + label
                    overlapped_obstacles.append(overlapped)
                elif overlapped.type in ['MultiPolygon', 'GeometryCollection']:
                    for p in overlapped:
                        if p.type == 'Polygon':
                            p.label = o_o.label + ',' + label
                            overlapped_obstacles.append(p)
                    # otherwise the new piece is wholly within overlapped

                # trim the shape to the unoverlapped sections
                shape = shape - o_o

            # anything that's hasn't overlapped can be added now
            if shape.type == 'Polygon':
                shape.label = label
                overlapped_obstacles.append(shape)
            elif shape.type in ['MultiPolygon', 'GeometryCollection']:
                for p in shape:
                    if p.type == 'Polygon':
                        p.label = label
                        overlapped_obstacles.append(p)

        self.overlapped_obstacles = overlapped_obstacles
        self._current = True


    def plot_obstacles(self, labels=False):
        """
        Outputs the original obstacles, without highlighting intersections
        """
        MCR.plot_shapes(self.obstacles, labels)
        MCR.plot_points([self.start, self.goal])


    def plot_overlapped_obstacles(self, labels=False):
        """
        Outputs the obstacles, including intersections
        """
        if not self._current:
            self.construct_overlaps()

        MCR.plot_shapes(self.overlapped_obstacles, labels)
        MCR.plot_points([self.start, self.goal])


    @staticmethod
    def plot_points(points):
        """
        Plots shapely Point objects
        """
        xs = [pt.x for pt in points]
        ys = [pt.y for pt in points]
        plt.scatter(xs, ys, **MCR.point_opts)


    @staticmethod
    def plot_linestrings(linestrings):
        """
        Plots shapely linestrings objects
        """
        for linestring in linestrings:
            for line in zip(linestring.coords, linestring.coords[1:]):
                plt.plot(*zip(*line)) # I know...


    @staticmethod
    def plot_shapes(shapes, labels=False):
        """
        Plots shapely Polygon objects (the exterior only)
        """
        # TODO: change points param into a graph?
        # add obstacles
        for s in shapes:
            opts = MCR.shape_opts.copy()
            if hasattr(s, 'facecolor'):
                opts['facecolor'] = s.facecolor
            if hasattr(s, 'edgecolor'):
                opts['edgecolor'] = s.edgecolor
            poly = plt.Polygon(s.exterior.coords, **opts)
            plt.gca().add_patch(poly)

            if labels and hasattr(s, 'label'):
                r_p = s.representative_point()
                plt.text(r_p.x, r_p.y, s.label,
                         horizontalalignment='center',
                         verticalalignment='center')


    def intersections_of(self, label):
        """
        Returns a list of the other obstacles which intersect this one
        """
        if not self._current:
            self.construct_overlaps()

        prepped = prep(self.obstacles[label - 1])
        return [o_o for o_o in self.obstacles if prepped.intersects(o_o)]

    def create_graph(self, labels=False):
        '''
        Create the intersection graph. This should be done anytime an object is
        added or removed, and whenever the start/goal locations change.
        '''
        if not self._current:
            self.construct_overlaps()

        sections = self.overlapped_obstacles[:]

        # label and append whitespace to polygons in sections
        field = Polygon(self.field)
        for o in self.obstacles:
            field -= o

        if field.type == 'Polygon':
            field.label = ''
            sections.append(field)
        else:
            for f in field:
                f.label = ''
                sections.append(f)

         # refresh current graph
        self.graph.clear()
        self.node_labels = {}
        self.node_pos = {}

        for section in sections:
            # Polygons aren't hashable -- but their well-known texts are
            wkt = section.wkt
            self.graph.add_node(wkt)
            pt = section.representative_point()
            self.node_pos[wkt] = (pt.x, pt.y)
            self.node_labels[wkt] = section.label
            # Do these overlap in a line(s) (1-D)? Use the DE-9IM relationship:
            # http://giswiki.hsr.ch/images/3/3d/9dem_springer.pdf
            adjacencies = [x for x in sections if section.relate(x)[4] == '1']
            for adj in adjacencies:
                self.graph.add_edge(wkt, adj.wkt)

        # Add start and goal
        self.graph.add_node(self.start.wkt)
        self.node_pos[self.start.wkt] = (self.start.x, self.start.y)
        self.node_labels[self.start.wkt] = 'start'

        self.graph.add_node(self.goal.wkt)
        self.node_pos[self.goal.wkt] = (self.goal.x, self.goal.y)
        self.node_labels[self.goal.wkt] = 'goal'

        for section in sections:
            if section.contains(self.start):
                self.graph.add_edge(self.start.wkt, section.wkt)
            if section.contains(self.goal):
                self.graph.add_edge(self.goal.wkt, section.wkt)

    def plot_graph(self, labels=False):
        """
        Outputs the square
        """
        if not self._current:
            self.construct_overlaps()
            self.create_graph()

        nx.draw_networkx(self.graph,
                         self.node_pos,
                         with_labels=False,
                         **MCR.nx_opts)

        if labels:  # offset labels below nodes
            label_pos = {node: (x + 0.025, y - 0.025)
                         for node, (x, y) in self.node_pos.items()}
            nx.draw_networkx_labels(self.graph,
                                    label_pos,
                                    labels=self.node_labels,
                                    **MCR.nx_opts)

    def setup_axes(self, **tick_params):
        """
        Set up plotting axis (black outline, no tickmarks)
        """
        if tick_params:
            params = tick_params
        else:
            params = {'axis': 'both', # changes apply to the x-axis
                      'which': 'both', # major and minor ticks are affected
                      # ticks along the bottom edge are off
                      'top': 'off', # ticks along the top edge are off
                      'bottom': 'off', # ditto
                      'left': 'off', # ditto
                      'right': 'off', # ditto
                      'labelbottom': 'off',
                      'labelleft': 'off'}
        plt.axis([0, 1, 0, 1])
        plt.tick_params(**params)  # labels along the bottom edge are off

    def show(self, obstacles=True, graph=True, labels=False):
        """
        The default display method. If you want to show
        """
        if obstacles:
            self.plot_obstacles(labels=(not graph))

        if graph:
            self.plot_graph(labels=labels)

        self.setup_axes()
        plt.show()

    def svg(self):
        """
        Draws an SVG
        """
        # TODO: I believe all I'll need is to change the output method of
        # matplotlib, then redraw.
        # Do I need an output file?
        pass

    # Helper methods

    def __parse_SVG(svg_file):
        try:
            f = open(svg_file)
            svg = f.read()
            f.close()
        except FileNotFoundError:
            print('File {} not found'.format(svg_file))
            return

        shapes = []

        #  Viewbox - this will be scaled to 1, 1 eventually
        try:
            viewbox = [float(x)
                       for x in re.findall('viewBox="(.*?)"', svg)[0].split()]
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

            # Transforms
            transform = re.findall('transform="(.*?)"', r)
            if transform:
                t_list = re.findall('\w+\(.+?\)', transform[0])

                # reverse to accumulate transform composion
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
                        scale_factors = [float(x)
                                         for x in re.split('\s+,?\s*', vals)]
                        # convert 1-arg shorthand to full 2-arg version
                        if len(scale_factors) == 1:
                            scale_factors[1] = scale_factors[0]
                        rect = scale(rect, *scale_factors, origin=(0, 0))

                    elif t.startswith('rotate'):
                        val = re.findall('\((.+)\)', t)[0]
                        rect = rotate(rect, float(val), origin=(0, 0))

                    elif t.startswith('skewX'):
                        val = re.findall('\((.+)\)', t)[0]
                        rect = skew(rect, xs=float(vals), origin=(0, 0))

                    elif t.startswith('skewY'):
                        val = re.findall('\((.+?)\)', t)[0]
                        rect = skew(rect, ys=float(vals), origin=(0, 0))

            shapes.append(rect)

        # Polygons
        polygons = re.findall('<polygon.*?\/>', svg)
        for p in polygons:
            polygon = []
            point_list = re.findall('points="(.*?)"', p)
            points = ([float(x) for x in point_list[0].split()])

            points.reverse()  # so that I can pop in x,y order
            while points:
                polygon.append((points.pop(), points.pop()))

            # remove the doubled last point
            polygon = polygon[:-1]

            shapes.append(Polygon(polygon))

        # rescale to [1,1]
        scaled_shapes = []
        for shape in shapes:
            # N.b.: the svg viewbox starts at the top left
            shape = translate(scale(shape,
                                    xfact=1 / vb_w,
                                    yfact=-1 / vb_h,
                                    origin=(0, 0)),
                              0, 1)
            scaled_shapes.append(shape)

        return scaled_shapes


def random_MCR(obstacles=10, scale_factor=0.25):
    """
    Creates an MCR with random polygonal obstacles
    """
    mcr = MCR()
    pts = poisson_2d(obstacles)
    new_obs = []

    for i in range(obstacles):
        r = scale(random_shape(), scale_factor, scale_factor)
        x_min, y_min, _, _ = r.bounds
        t_x, t_y = pts.pop()
        new_obs.append(translate(r, t_x - x_min, t_y - y_min))

    # rescale again, to fit all shapes in the
    bounds = [o.bounds for o in new_obs]
    max_x = max([b[2] for b in bounds])
    max_y = max([b[3] for b in bounds])

    for obstacle in new_obs:
        mcr.add_obstacle(scale(obstacle, 1 / max_x, 1 / max_y, origin=(0, 0)))

    return mcr


def random_shape(max_sides=6):
    """
    Creates a random convex polygon with between 3 and ~max_sides sides all of
    which have length ~1.
    """
    return rotate(approx_ngon(rand.randrange(3, max_sides + 1)),
                  rand.uniform(0, 360),
                  origin='centroid')


def on_left_hand(point, ray):
    """
    Is point on the left-hand side of the directed ray ray[0] - ray[1]?
    This includes being on the line itself.
    """
    px, py = point
    x1, y1 = ray[0]
    x2, y2 = ray[1]

    # special case for vertical lines (no slope)
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
    length 1. It's guaranteed that the polygon will be convex.
    """
    ngon = [(0, 0), (1, 0)]  # the polygon
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
    Returns an array of k (x, y) tuples evenly spaced across [0, 1] x [0, 1]
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
