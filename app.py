from numpy import array, sum, uint8
from skimage import io, color
import cv2
from flask import Flask, request, render_template, send_file
from numpy import array, sqrt
import heapq
from PIL import Image, ImageDraw, ImageOps
import io


class ImageSeg:
    def __init__(self, path):
        self.path = path
        self.img = io.imread(path)
        self.threshold = 0

    def visualize_rgb(self):
        rgb_img = self.img
        img = Image.fromarray((rgb_img * 255).astype(uint8))  # Convert float image to 8-bit for display
        img.show()

    def RGNull(self):
        arr = array(self.img)
        greenval = 0
        count = 0
        for i in range(len(arr)):
            for j in range(len(arr[i])):
                count += 1
                greenval += arr[i][j][1]
                arr[i][j][0] = 0
                arr[i][j][2] = 0
        self.threshold = (greenval / count) / 1.5
        return arr

    def IsoGray(self):
        RGNull_img = self.RGNull()
        gray_img = cv2.cvtColor(RGNull_img, cv2.COLOR_RGB2GRAY)
        return gray_img

    def IsoGrayThresh(self):
        gray_img = self.IsoGray()
        thresh_img = (gray_img > self.threshold) * 255
        img = Image.fromarray(thresh_img.astype(uint8))  # Convert array to image
        img.show()
        return thresh_img

    def visualize_compare(self):
        rgb_img = Image.fromarray((self.img * 255).astype(uint8))
        gray_img = Image.fromarray(self.IsoGray().astype(uint8))
        thresh_img = Image.fromarray(self.IsoGrayThresh().astype(uint8))

        rgb_img.show(title="RGB Image")
        gray_img.show(title="Grayscale Image")
        thresh_img.show(title="Thresholded Image")

    def PixelCount(self):
        count = 0
        arr = self.IsoGrayThresh()
        count = sum(arr != 0)
        return count


class OptimalPathing:
    def __init__(self, img):
        self.img = array(img)

    def create_graph(self, binary_image):
        rows, cols = binary_image.shape
        graph = {}
        neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Only 4 neighbors for simplicity

        for i in range(rows):
            for j in range(cols):
                neighbors = []
                for dx, dy in neighbor_offsets:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < rows and 0 <= nj < cols:
                        weight = sqrt((ni - rows + 1) ** 2 + (nj - cols + 1) ** 2) + (255 - binary_image[ni, nj])
                        neighbors.append(((ni, nj), weight))
                graph[(i, j)] = neighbors
        return graph

    def trace_path(self, parents, start, target):
        path = []
        current = target
        while current != start:
            path.append(current)
            current = parents[current]
        path.append(start)
        path.reverse()
        return path

    def ComputeDjikstra(self, start_pixel=(0, 0), target_pixel=None):
        if target_pixel is None:
            target_pixel = (self.img.shape[0] - 1, self.img.shape[1] - 1)
        graph = self.create_graph(self.img)
        parents = {}
        heap = [(0, start_pixel)]
        visited = set()

        while heap:
            cost, current = heapq.heappop(heap)

            if current in visited:
                continue

            visited.add(current)

            if current == target_pixel:
                break

            for neighbor, weight in graph.get(current, []):
                if neighbor not in visited:
                    parents[neighbor] = current
                    heapq.heappush(heap, (cost + weight, neighbor))

        shortest_path = self.trace_path(parents, start_pixel, target_pixel)

        # Visualize the image and the shortest path using PIL
        img = Image.fromarray(self.img)
        img_color = ImageOps.colorize(img.convert('L'), 'black', 'white')

        draw = ImageDraw.Draw(img_color)
        if shortest_path:
            for i in range(len(shortest_path) - 1):
                start_point = (shortest_path[i][1], shortest_path[i][0])
                end_point = (shortest_path[i + 1][1], shortest_path[i + 1][0])
                # Ensure path does not go out of bounds
                start_point = (max(0, min(img_color.width - 1, start_point[0])), max(0, min(img_color.height - 1, start_point[1])))
                end_point = (max(0, min(img_color.width - 1, end_point[0])), max(0, min(img_color.height - 1, end_point[1])))
                draw.line([start_point, end_point], fill="blue", width=2)

        # Save the result to a BytesIO object
        img_bytes = io.BytesIO()
        img_color.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        return img_bytes


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    if file:
        image = Image.open(file).convert('L')  # Convert to grayscale
        img_array = array(image)
        obj = OptimalPathing(img_array)
        result_img = obj.ComputeDjikstra()
        return send_file(result_img, mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)