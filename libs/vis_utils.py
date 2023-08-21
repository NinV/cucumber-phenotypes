from typing import Union
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Point, LineString, Polygon


def draw_point(image, point:Union[Point, tuple], color=(255, 0, 0), size=3):
    """
    Draw a point on the image.

    Args:
        image (PIL.Image.Image): The image to draw on.
        point (shapely.geometry.Point or (2, ) array-like): The point geometry object.
        color (tuple, optional): RGB color values (default: red).
        size (int, optional): The size of the point (default: 3).
    """
    draw = ImageDraw.Draw(image)
    if isinstance(point, Point):
        x, y = point.x, point.y
    else:
        x, y = point
    draw.ellipse([(x - size, y - size), (x + size, y + size)], fill=color)


def draw_line(image, lines, color=(0, 255, 0), width=2):
    """
    Draw a line on the image.

    Args:
        image (PIL.Image.Image): The image to draw on.
        lines (shapely.geometry.LineString or [(x, y), (x, y), ...]): The line geometry object.
        color (tuple, optional): RGB color values (default: green).
        width (int, optional): The width of the line (default: 2).
    """
    draw = ImageDraw.Draw(image)
    if isinstance(lines, LineString):
        xy = list(lines.coords)
        draw.line(xy, fill=color, width=width)
    else:
        for xy1, xy2 in zip(lines[:-1], lines[1:]):
            draw.line([tuple(xy1), tuple(xy2)], fill=color, width=width)


def draw_polygon(image, polygon, fill_color=(0, 0, 255), outline_color=(255, 255, 255), width=2):
    """
    Draw a polygon on the image.

    Args:
        image (PIL.Image.Image): The image to draw on.
        polygon (shapely.geometry.Polygon): The polygon geometry object.
        fill_color (tuple, optional): RGB color values for the polygon fill (default: blue).
        outline_color (tuple, optional): RGB color values for the polygon outline (default: white).
        width (int, optional): The width of the outline (default: 2).
    """
    draw = ImageDraw.Draw(image)
    xy = list(polygon.exterior.coords)
    draw.polygon(xy, fill=fill_color, outline=outline_color, width=width)


def draw_text(image, location, text, color=(0, 255, 0), font_type='UbuntuMono-BI.ttf', font_size=32):
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_type, size=font_size)
    draw.text(location, text, fill=color, font=font)
