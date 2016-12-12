### Data Processing ###

def convert_to_old_representation(new_data):
    return new_data

def set_to_buckets(parsed_data, buckets_per_continuous):
    standard_data = []
    schema = parsed_data[0].schema
    min_data = {}
    max_data = {}
    for item in parsed_data:
        for attribute in schema:
            if attribute.lower() in EXCLUDE_THESE:
                continue
            if item.get(attribute).type == 'CONTINUOUS':
                if attribute not in min_data:
                    min_data[attribute] = item.get(attribute).value
                min_data[attribute] = min(min_data[attribute], item.get(attribute).value)
                if attribute not in max_data:
                    max_data[attribute] = item.get(attribute).value
                max_data[attribute] = max(max_data[attribute], item.get(attribute).value)

    bucket_params = (buckets_per_continuous, min_data, max_data,)
    for item in parsed_data:
        for attribute in min_data:
            bucket = what_is_my_bucket(item.get(attribute).value, attribute, bucket_params)
            old_entry = item.get(attribute)
            item.set(attribute, Entry(old_entry.index, old_entry.name, 'BUCKET CONTINUOUS', buckets_per_continuous, bucket))

        for attribute in item.schema:
            if isinstance(item.get(attribute).value, str):
                old_entry = item.get(attribute)
                new_value = old_entry.possible_values.index(old_entry.value)
                item.set(attribute, Entry(old_entry.index, old_entry.name, 'BUCKET NOMINAL', len(old_entry.possible_values), new_value))
        standard_data.append(item)

    return standard_data, bucket_params

def run_algorithm(
    dataset,
    split=0.9,
    cross_validation=True,
    num_buckets=10,
    m_estimate=0,
    debug_output=False
    ):
    if num_continuous_buckets <= 2:
        num_continuous_buckets = 2

    if debug_output:
        print('Begin parsing data')
        parsed_data = dataset
        print('Done parsing data')

    data, bucket_params = set_to_buckets(parsed_data, num_continuous_buckets)

    EXCLUDE_THESE = ['class']

    num_input_units = 0
    for item in data[0].schema:
        if item.lower() in EXCLUDE_THESE:
            continue
        else:
            num_input_units += 1

    if not cross_validation:
        if debug_output:
            print('no cross validation')
        training_data = data._training
        test_data = data[int(floor(len(data) * split)):]

        nab = NaiveBayes(num_input_units, m_estimate, debug_output)
        nab.train(training_data)
        result = nab.classify(test_data, bucket_params)
        result.print_output()
    else: # cross_validation == True
