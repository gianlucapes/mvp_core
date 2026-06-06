from pinn_heat.domain.plate2d import Plate2d


def main():
    plate = Plate2d(width_mm=100.0, height_mm=100.0, t_hot=373.15, t_amb=293.15)
    print("Hello from mvp-core!")


if __name__ == "__main__":
    main()
