#===============================================================================
# Exemplo: segmentação de uma imagem em escala de cinza.
#-------------------------------------------------------------------------------
# Autor: Bogdan T. Nassu
# Universidade Tecnológica Federal do Paraná
#===============================================================================

import sys
import timeit
import numpy as np
import cv2

#===============================================================================

INPUT_IMAGE =  'arroz.bmp'

# TODO: ajuste estes parâmetros!
NEGATIVO = False
THRESHOLD = 0.8
ALTURA_MIN = 20
LARGURA_MIN = 20
N_PIXELS_MIN = 10

#===============================================================================

def binariza (img, threshold):
    #img_binarizada = np.where(img > threshold, 1, 0).astype(np.uint8)
    for i in range (len(img)):
        for j in range(len(img[0])):
            if img[i,j]>threshold:
                img[i][j]=1
            else:
                img[i][j]=0     
    return img
  
def rotula(img, largura_min, altura_min, n_pixels_min):
    rows, cols = img.shape[:2]
    label =0
    componentes_rotulados=[]

    def inunda(label,x,y):
        if x < 0 or x >= rows or y < 0 or y >= cols or img[x,y] != 1:
            return {'label': label, 'n_pixels': 0, 'T': x, 'L': y, 'B': x, 'R': y}

        img[x, y] = label
        return {'label': label,'n_pixels': 1,'T': x,'L': y,'B': x,'R': y,**inunda(label,x + 1, y),
        **inunda(label,x - 1, y),
        **inunda(label,x, y + 1),
        **inunda(label,x, y - 1),
        }


    for i in range(rows):
        for j in range(cols):
            if img[i,j]==1:
                label+=0.1
                componente=inunda(label,i,j)
                if(componente['n_pixels'] >= n_pixels_min and componente['B'] - componente['T'] + 1 >= altura_min 
                and componente['R'] - componente['L'] + 1 >= largura_min):
                    componentes_rotulados.append(componente)
    return componentes_rotulados

def main ():

    # Abre a imagem em escala de cinza.
    img = cv2.imread (INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()

    # É uma boa prática manter o shape com 3 valores, independente da imagem ser
    # colorida ou não. Também já convertemos para float32.
    img = img.reshape ((img.shape [0], img.shape [1], 1))
    img = img.astype (np.float32) / 255

    # Mantém uma cópia colorida para desenhar a saída.
    img_out = cv2.cvtColor (img, cv2.COLOR_GRAY2BGR)

    # Segmenta a imagem.
    if NEGATIVO:
        img = 1 - img
    img = binariza (img, THRESHOLD)
    cv2.imshow ('01 - binarizada', img)
    cv2.imwrite ('01 - binarizada.png', img*255)

    start_time = timeit.default_timer ()
    componentes = rotula (img, LARGURA_MIN, ALTURA_MIN, N_PIXELS_MIN)
    n_componentes = len (componentes)
    print ('Tempo: %f' % (timeit.default_timer () - start_time))
    print ('%d componentes detectados.' % n_componentes)

    # Mostra os objetos encontrados.
    for c in componentes:
        cv2.rectangle (img_out, (c ['L'], c ['T']), (c ['R'], c ['B']), (0,0,1))

    cv2.imshow ('02 - out', img_out)
    cv2.imwrite ('02 - out.png', img_out*255)
    cv2.waitKey ()
    cv2.destroyAllWindows ()


if __name__ == '__main__':
    main ()

#===============================================================================
