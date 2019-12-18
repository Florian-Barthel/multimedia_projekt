class AnnotationRect:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = min((x1, x2))
        self.x2 = max((x1, x2))
        self.y1 = min((y1, y2))
        self.y2 = max((y1, y2))

    def __str__(self):
        return 'x1: {}, y1: {}, x2: {}, y2: {}'.format(self.x1, self.y1, self.x2, self.y2)

    def width(self):
        return self.x2 - self.x1

    def height(self):
        return self.y2 - self.y1

    def area(self):
        return self.width() * self.height()
        