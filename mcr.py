from math import *
import shapely as sh
from shapely.geometry import *
from shapely.affinity import *
import graphviz as gv
import random as rand
import matplotlib.pyplot as plt
import re


class mcr:

    """
    Minimum Cover Removal. Contains a number of helper methods for
    investigating the MCR problem as described in Erickson and LaValle 2013 and
    Hauser 2012
    """

    # We display obstacles by default as light gray with visible overlapping
    # The start and goal points are red
    shape_opts = {'alpha': 0.25, 'edgecolor': 'black', 'facecolor': 'gray'}
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
        # Shapes are labeled--if not explicitly, then with this
        self.__label = 0

        # self.field = plt.Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])

        if svg:  # initialize from svg file
            try:
                obstacles = mcr.__parse_SVG(svg)
                for o in obstacles:
                    self.add_obstacle(o)
            except:
                print('Couldn\'t load all shapes in ', svg)
                raise ValueError

    def add_obstacle(self, shape):
        """
        Adds new obstacles to the field. They'll be added to self.obstacles
        and self.overlapped_obstacles.
        """
        if not hasattr(shape, 'label'):
            new_label = str(self.__label)
            shape.label = new_label
            self.__label += 1
        else:
            new_label = shape.label

        self.obstacles.append(shape)

        # for every shape that is added, it may intersect with any other
        overlaps = [
            o_o for o_o in self.overlapped_obstacles if o_o.intersects(shape)]

        if len(overlaps) == 0:  # quick escape
            self.overlapped_obstacles.append(shape)
            return

        for o_o in overlaps:
            # first remove, then re-add piece by piece
            self.overlapped_obstacles.remove(o_o)
            unoverlapped_part = o_o - shape
            overlapped_part = o_o & shape

            # we don't care about Points, Lines, etc.
            if unoverlapped_part.type == 'Polygon':
                unoverlapped_part.label = o_o.label
                self.overlapped_obstacles.append(unoverlapped_part)
            elif unoverlapped_part.type in ['MultiPolygon', 'GeometryCollection']:
                for p in unoverlapped_part:
                    if p.type == 'Polygon':
                        p.label = o_o.label
                        self.overlapped_obstacles.append(p)

            # otherwise the new piece is wholly within unoverlapped_part
            if overlapped_part.type == 'Polygon':
                overlapped_part.label = o_o.label + ',' + new_label
                self.overlapped_obstacles.append(overlapped_part)
            elif overlapped_part.type in ['MultiPolygon', 'GeometryCollection']:
                for p in overlapped_part:
                    if p.type == 'Polygon':
                        p.label = o_o.label + ',' + new_label
                        self.overlapped_obstacles.append(p)
                # otherwise the new piece is wholly within overlapped_part

            # trim the shape to the unoverlapped sections
            shape = shape - o_o

        # anything that's hasn't overlapped can be added now
        if shape.type == 'Polygon':
            shape.label = new_label
            self.overlapped_obstacles.append(shape)
        elif shape.type in ['MultiPolygon', 'GeometryCollection']:
            for p in shape:
                if p.type == 'Polygon':
                    p.label = new_label
                    self.overlapped_obstacles.append(p)

    def remove_obstacle(self, label):
        """
        Removes the obstacle labeled 'label'.
        Removes the labeled vertices from the graph and contracts those edges
        """
        # N.B.: numbering starts at 1
        self.obstacles = self.obstacles[:label-1] + self.obstacles[label:]

        self.overlapped_obstacles = filter(lambda x:
                                           x.label.find(str(label) != -1),
                                           self.overlapped_obstacles)
        # TODO: change the graph

    def show_bare_obstacles(self):
        """
        Outputs the original obstacles, without highlighting intersections
        """
        mcr.__plot_shapes(self.obstacles)
        mcr.__plot_points([self.start, self.goal])
        plt.axis([0, 1, 0, 1])
        plt.show()

    def show_obstacles(self, recreate=True):
        """
        Outputs the obstacles, includin intersections
        """
        plt.axis([0, 1, 0, 1])
        mcr.__plot_shapes(self.overlapped_obstacles)
        mcr.__plot_points([self.start, self.goal])
        plt.show()

    def __plot_shapes(shapes):
        # TODO: change points param into a graph?
        # add obstacles
        for s in shapes:
            opts = mcr.shape_opts.copy()
            if hasattr(s, 'facecolor'):
                opts['facecolor'] = s.facecolor
            if hasattr(s, 'edgecolor'):
                opts['edgecolor'] = s.edgecolor
            poly = plt.Polygon(s.exterior.coords, **opts)
            plt.gca().add_patch(poly)

            if hasattr(s, 'label'):
                r_p = s.representative_point()
                plt.text(r_p.x, r_p.y, s.label,
                         horizontalalignment='center',
                         verticalalignment='center')
            if __debug__:
                print(s.label + ": " + str(s.area))

    def __plot_points(points):
        # add start and goal
        xs = [x for x, _ in points]
        ys = [y for _, y in points]

        plt.scatter(xs, ys, **mcr.point_opts)

    def create_graph(self):
        """
        Create the intersection graph. This should be done anytime an object is
        added or removed, and whenever the start/goal locations change.
        Note that this is called by default every time show_obstacles() is run.
        """
        pass

    def show_graph(self):
        """
        Outputs the square
        """
        gv.Source(self.graph)

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
                        scale_factors = [float(x) for x in re.split('\s+,?\s*', vals)]
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


def random_mcr(obstacles=10, scale_factor=0.25):
    """
    Creates an MCR with random polygonal obstacles
    """
    mcr = mcr()
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
