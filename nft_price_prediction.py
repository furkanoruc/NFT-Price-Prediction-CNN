from os.path import join

from visual_feature_extractor import VisualFeatureExtractor


def main(images_path: str, prices_path: str, number_of_epochs: int, batch_size: int):
    feature_extractor = VisualFeatureExtractor(
        images_path,
        prices_path,
        number_of_epochs,
        batch_size,
        image_size=(256, 256),
        train_ratio=.6,
        validation_ratio=.2
    )
    feature_extractor.run()
    print(f'Mean squared error: {feature_extractor.mean_squared_error}')
    print(f'Features\' shape: {feature_extractor.features.shape}')


if __name__ == '__main__':
    main(images_path=join('..', 'data', 'images'),
         prices_path=join('..', 'data', 'nft.pickle'),
         number_of_epochs=10,
         batch_size=128)
