from scripts.publish_results_to_kaggle import kaggle_environment


def test_kaggle_key_is_only_passed_in_environment():
    environment = kaggle_environment("owner", "secret-value")
    assert environment["KAGGLE_USERNAME"] == "owner"
    assert environment["KAGGLE_KEY"] == "secret-value"
    assert environment["KAGGLE_CONFIG_DIR"].endswith("kaggle_runtime")
