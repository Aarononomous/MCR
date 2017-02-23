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
    shape_opts = {'alpha': 0.15, 'edgecolor': 'black', 'facecolor': 'gray'}
    point_opts = {'color': 'red'}

    def __init__(self, svg=None):
        """
        Create an empty square to add shapes to.
        """
        self.obstacles = []
        self.overlapped_obstacles = []
        self.graph = []
        self.start = (0.15, 0.05)
        self.goal = (0.95, 0.95)
        self._label = 1  # Shapes are labeled either explicitly or using this
        self._current = True  # False if overlaps need to be recalculated

        self.field = plt.Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])

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
            prepped = prep(shape) # speed up testing
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

    def show_bare_obstacles(self, labels=False):
        """
        Outputs the original obstacles, without highlighting intersections
        """
        MCR.__plot_shapes(self.obstacles, labels)
        MCR.__plot_points([self.start, self.goal])
        plt.axis([0, 1, 0, 1])
        plt.show()

    def show_obstacles(self, labels=False):
        """
        Outputs the obstacles, including intersections
        """
        if not self._current:
            self.construct_overlaps()

        plt.axis([0, 1, 0, 1])
        MCR.__plot_shapes(self.overlapped_obstacles, labels)
        MCR.__plot_points([self.start, self.goal])
        plt.show()

    def __plot_shapes(shapes, labels):
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
            # if __debug__:
            #     print(s.label + ": " + str(s.area))

    def __plot_points(points):
        xs = [x for x, _ in points]
        ys = [y for _, y in points]
        plt.scatter(xs, ys, **MCR.point_opts)

    def intersections_of(self, label):
        """
        Returns a list of the other obstacles which intersect this one
        """
        prepped = prep(self.obstacles[label - 1])
        return [o_o for o_o in self.overlapped_obstacles if prepped.intersects(o_o)]


    def create_graph(self):
        '''
        Create the intersection graph thingy. This should be done anytime an object is
        added or removed, and whenever the start/goal locations change.
        Note that this is called by default every time show_obstacles() is run.
        '''
        #gather centroids for vertices
        centroids = {}
        
        for o in self.overlapped_obstacles:
            centroids[o.label] = (o.centroid.x, o.centroid.y)
            
        centroids['start'] = self.start
        centroids['goal'] = self.goal
        
        #create graph
        G = nx.Graph()
        
        G.add_nodes_from(centroids.keys())
        
        #save node attributes position and labels
        for n, p in centroids.items():
            G.node[n]['pos'] = p
            G.node[n]['label'] = n
            
        pos = nx.get_node_attributes(G, 'pos')
        labels = nx.get_node_attributes(G, 'label')
        
        #determine edges between obstacle nodes and intersection nodes
        #note: assuming no edges exist between intersections and intersections (?)
        edges = []
        for x, y in combinations(enumerate(self.overlapped_obstacles),2):
            label = x[1].label + y[1].label
            if x[1].intersects(y[1]) and label.count(",") >= 1: 
                edges.append((x[1].label, y[1].label))
                        
        #note: assuming no edges exist between intersection nodes and white space (?)
        #note: insert useful thing here plz
        whitespace_edges = self.get_whitespace_edges()
        
        G.add_edges_from(edges)
        G.add_edges_from(whitespace_edges)
        
        nx.draw(G, pos)
        plt.show()
        
        return G
        
    def get_whitespace_edges(self):
        '''
        Ideally this will be a smarter method. What I would like 
        is whitespace polygons that I can run the same snippet of code as is
        in create_graph
        '''
        end_edges = []
        #whitespace_edges = []
        #relevant_obstacles = whitespace_obstacles + obstacles
        #for x, y in combinations(enumerate(relevant_obstacles),2):
            #label = x[1].label + y[1].label
            #if x[1].intersects(y[1]) and label.count(",") >= 1: 
                #whitespace_edges.append((x[1].label, y[1].label))
                
        #doing something dumb in the meantime
        for o in self.obstacles:
            end_edges.append(('start', o.label))
            end_edges.append(('goal', o.label))
        
        return end_edges

    def create_graph_aaj(self):
        """
        Create the intersection graph. This should be done anytime an object is
        added or removed, and whenever the start/goal locations change.
        Note that this is called by default every time show_obstacles() is run.
        """
        # add background field to overlapped_obstacles
        field = Polygon([(0,0), (1,0), (1,1), (0,1)])
        for o in f.obstacles:
            field -= o
        sections = f.overlapped_obstacles + field

        g = Graph()
        for section in sections:
            g.add_vertex(section.label, section.representative_point)
            adjacencies = [x for x in sections if section.touches(x)]
            for adj in adjacencies:
                g.add_edge(section.label, adj.label)
        return g

    def show_graph(self):
        """
        Outputs the square
        """
        if not self._current:
            self.construct_overlaps()

        g = self.create_graph()
        MCR.__plot_points(g.vertices.values())
        for v in g.adj:
            for w in g.adj[v]:
                    if (v <= w):
                        line = plt.Line2D(g.vertices[v], g.vertices[w])
                        plt.gca().add_line(line)
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

    print('Done!')
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
