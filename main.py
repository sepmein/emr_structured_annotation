def main():
    print("Hello from emr-labelling!")
    from gliner2 import GLiNER2

    # Load model once, use everywhere
    extractor = GLiNER2.from_pretrained("fastino/gliner2-base-v1")

    # Check what backbone the loaded checkpoint declares
    print("model_name:", getattr(extractor.config, "model_name", None))
    print("counting_layer:", getattr(extractor.config, "counting_layer", None))
    print("token_pooling:", getattr(extractor.config, "token_pooling", None))

    # Extract entities in one line
    text = "Apple CEO Tim Cook announced iPhone 15 in Cupertino yesterday."
    result = extractor.extract_entities(
        text, ["company", "person", "product", "location"]
    )

    print(result)
    # {'entities': {'company': ['Apple'], 'person': ['Tim Cook'], 'product': ['iPhone 15'], 'location': ['Cupertino']}}


if __name__ == "__main__":
    main()
