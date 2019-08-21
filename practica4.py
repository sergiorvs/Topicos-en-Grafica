from PIL import Image
import cmath
pi2 = cmath.pi * 2.0

G = [[1, 4, 7, 4, 1],
     [4, 16, 26, 16, 4],
     [7, 26, 41, 26, 7],
     [4, 16, 26, 16, 4],
     [1, 4, 7, 4, 1]]


def DFT2D(image):
    global M, N
    (M, N) = image.size  # (imgx, imgy)
    dft2d_gray = [[0.0 for k in range(M)] for l in range(N)]
    """dft2d_grn = [[0.0 for k in range(M)] for l in range(N)]
    dft2d_blu = [[0.0 for k in range(M)] for l in range(N)]"""
    pixels = image.load()
    #print(pixels[0,0])
    for k in range(M):
        for l in range(N):
            sum_gray = 0.0
            #sum_grn = 0.0
            #sum_blu = 0.0
            for m in range(M):
                for n in range(N):
                    #print('m', m, n)
                    try:
                        (gray) = pixels[m, n]             
                    except IndexError:
                        (gray) = 0
                    e = cmath.exp(- 1j * pi2 *
                                  (float(k * m) / M + float(l * n) / N))
                    sum_gray += e * gray
                    #sum_grn += grn * e
                    #sum_blu += blu * e
            dft2d_gray[l][k] = sum_gray / M / N
            """dft2d_grn[l][k] = sum_grn / M / N
            dft2d_blu[l][k] = sum_blu / M / N"""
    # dft2d_gray.show()
    return (dft2d_gray)  # , dft2d_grn, dft2d_blu)


def IDFT2D(dft2d):
    #(dft2d_red, dft2d_grn, dft2d_blu) = dft2d
    #dft2d
    global M, N
    #sum_grn = 0.0
    image = Image.new("L", (M, N))
    pixels = image.load()
    for m in range(M):
        for n in range(N):
            sum_gray = 0.0
            #sum_grn = 0.0
            #sum_blu = 0.0
            for k in range(M):
                for l in range(N):
                    e = cmath.exp(
                        1j * pi2 * (float(k * m) / M + float(l * n) / N))
                    sum_gray += dft2d[l][k] * e
                    #sum_grn += dft2d_grn[l][k] * e
                    #sum_blu += dft2d_blu[l][k] * e
            gray = int(sum_gray.real + 0.5)
            #grn = int(sum_grn.real + 0.5)
            #blu = int(sum_blu.real + 0.5)
            pixels[m, n] = (gray)  # , grn, blu)
    image.show()
    return image


# TEST
# Recreate input image from 2D DFT results to compare to input image
Pimg = Image.open("/home/sergio/Descargas/mrb3.jpg").convert('L')
image = IDFT2D( DFT2D(Pimg) )
image.save("output3.png", "PNG")