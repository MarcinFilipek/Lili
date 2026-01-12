import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np


class BetterConvNet(nn.Module):
    def __init__(self, init_method="kaiming"):
        super(BetterConvNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=11, stride=2, padding=5)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=3)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=2)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(2, 2)

        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2)
        self.bn6 = nn.BatchNorm2d(512)

        self.relu = nn.ReLU()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # ZASTOSUJ WYBRANĄ INICJALIZACJĘ
        self._initialize_weights(init_method)

    def _initialize_weights(self, method):
        """
        Inicjalizuj wagi wybraną metodą.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if method == "kaiming":
                    # Domyślna (He initialization)
                    init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

                elif method == "xavier":
                    # Xavier/Glorot initialization
                    init.xavier_normal_(m.weight)

                elif method == "orthogonal":
                    # Orthogonal initialization
                    init.orthogonal_(m.weight)

                elif method == "sparse":
                    # Sparse initialization (większość wag = 0)
                    init.sparse_(m.weight, sparsity=0.9)

                elif method == "normal":
                    # Gaussian N(0, 0.01)
                    init.normal_(m.weight, mean=0, std=0.01)

                elif method == "uniform_small":
                    # Uniform w małym zakresie
                    init.uniform_(m.weight, -0.05, 0.05)

                elif method == "uniform_large":
                    # Uniform w dużym zakresie
                    init.uniform_(m.weight, -0.5, 0.5)

                elif method == "gabor":
                    # Gabor-like filters (próba symulacji V1)
                    self._init_gabor_like(m)

                elif method == "edge_detector":
                    # Ręczne filtry wykrywające krawędzie
                    self._init_edge_detectors(m)

                else:
                    raise ValueError(f"Unknown init method: {method}")

                # Bias zawsze zero
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _init_gabor_like(self, conv_layer):
        """
        Inicjalizuj filtry jako Gabor-like (podobne do V1 w mózgu).
        """
        n_filters = conv_layer.weight.shape[0]
        kernel_size = conv_layer.weight.shape[2]

        # Generuj Gabor filtry z różnymi orientacjami
        orientations = np.linspace(0, np.pi, n_filters)

        for i, theta in enumerate(orientations):
            # Parametry Gabor
            sigma = kernel_size / 6.0
            lambd = kernel_size / 2.0
            gamma = 0.5
            psi = 0

            # Generuj Gabor kernel
            gabor_kernel = self._create_gabor_kernel(
                kernel_size, sigma, theta, lambd, gamma, psi
            )

            # Przypisz do wszystkich kanałów wejściowych
            for c in range(conv_layer.weight.shape[1]):
                conv_layer.weight.data[i, c, :, :] = torch.from_numpy(gabor_kernel)

    def _create_gabor_kernel(self, size, sigma, theta, lambd, gamma, psi):
        """
        Tworzy Gabor kernel.
        """
        sigma_x = sigma
        sigma_y = sigma / gamma

        # Siatka współrzędnych
        center = size // 2
        x = np.arange(size) - center
        y = np.arange(size) - center
        X, Y = np.meshgrid(x, y)

        # Rotacja
        x_theta = X * np.cos(theta) + Y * np.sin(theta)
        y_theta = -X * np.sin(theta) + Y * np.cos(theta)

        # Gabor
        gb = np.exp(
            -0.5 * (x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2)
        ) * np.cos(2 * np.pi * x_theta / lambd + psi)

        return gb.astype(np.float32)

    def _init_edge_detectors(self, conv_layer):
        """
        Inicjalizuj jako klasyczne detektory krawędzi (Sobel, Prewitt, etc).
        """
        n_filters = conv_layer.weight.shape[0]
        kernel_size = conv_layer.weight.shape[2]

        if kernel_size == 3:
            # Sobel filters
            sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
            sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

            # Laplacian
            laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)

            filters = [sobel_x, sobel_y, laplacian]

            for i in range(min(n_filters, len(filters) * conv_layer.weight.shape[1])):
                filter_idx = i % len(filters)
                channel_idx = (i // len(filters)) % conv_layer.weight.shape[1]
                conv_layer.weight.data[i, channel_idx, :, :] = torch.from_numpy(
                    filters[filter_idx]
                )

        # Dla większych kerneli użyj random (TODO: lepsze filtry)
        else:
            init.xavier_normal_(conv_layer.weight)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)

        x = self.relu(self.bn5(self.conv5(x)))
        x = self.pool5(x)

        x = self.relu(self.bn6(self.conv6(x)))

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        return x
