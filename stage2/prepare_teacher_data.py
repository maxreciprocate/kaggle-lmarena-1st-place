# ;;
from datasets import load_dataset, Dataset, concatenate_datasets
from collections import Counter
from datasets import Features, Value
from transformers import AutoTokenizer
import hashlib
import datasets
datasets.disable_progress_bar()

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")

def cast(dataset):
    new_features = dataset.features.copy()
    for col in dataset.features:
        if dataset.features[col].dtype == 'string':
            new_features[col] = Value('large_string')
    return dataset.cast(new_features)

def hash_prompt_responses(x):
    prompt = x['prompt'][:512]
    response_a = x['response_a'][:512]
    response_b = x['response_b'][:512]
    if response_a > response_b:
        response_a, response_b = response_b, response_a
    return {"id": hashlib.sha256(f"{prompt}{response_a}{response_b}".encode()).hexdigest()}

def remove_response_duplicates(d):
    print(f"Before removing a == b: {len(d)}")
    d = d.filter(lambda x: x['response_a'].strip() != x['response_b'].strip())
    print(f"After remove a == b: {len(d)}")
    return d

def remove_duplicates(d):
    df = d.to_pandas()
    print(f"Before removing duplicates: {len(df)}")
    df = df.drop_duplicates(subset=["id"], keep="first")
    print(f"After removing duplicates: {len(df)}")
    return Dataset.from_pandas(df, preserve_index=False)

def balance_ab_winners_with_swap(d):
    print(f"Before balancing: {Counter(d['winner'])} {len(d)}")
    d_a = d.filter(lambda x: x['winner'] == 'model_a')
    d_b = d.filter(lambda x: x['winner'] == 'model_b')
    d_rest = d.filter(lambda x: x['winner'] not in ['model_a', 'model_b'])

    if len(d_b) > len(d_a):
        n_diff = (len(d_b) - len(d_a)) // 2
        d_b, d_b_to_swap = d_b.train_test_split(test_size=n_diff, seed=0).values()
        d_b_to_swap = d_b_to_swap.map(lambda x: {"winner": "model_a", "response_a": x["response_b"], "response_b": x["response_a"]})
        d_a = concatenate_datasets([d_a, d_b_to_swap])
    elif len(d_a) > len(d_b):
        n_diff = (len(d_a) - len(d_b)) // 2
        d_a, d_a_to_swap = d_a.train_test_split(test_size=n_diff, seed=0).values()
        d_a_to_swap = d_a_to_swap.map(lambda x: {"winner": "model_b", "response_a": x["response_b"], "response_b": x["response_a"]})
        d_b = concatenate_datasets([d_b, d_a_to_swap]).shuffle(seed=0)

    d = concatenate_datasets([d_a, d_b, d_rest])
    print(f"After balancing: {Counter(d['winner'])} {len(d)}")
    return d

def add_length(x):
    x["len_p"] = len(tokenizer(x["prompt"]).input_ids)
    x["len_a"] = len(tokenizer(x["response_a"]).input_ids)
    x["len_b"] = len(tokenizer(x["response_b"]).input_ids)
    x["len"] = x["len_p"] + x["len_a"] + x["len_b"]
    x["longer_won"] = x["len_a"] > x["len_b"] and x["winner"] == "model_a" or x["len_b"] > x["len_a"] and x["winner"] == "model_b"
    return x

def split_5_fold(d):
    folds = [
        (
            [i for i in range(len(d)) if i % 5 != fold],
            [i for i in range(len(d)) if i % 5 == fold]
        )
        for fold in range(5)

    ]
    train, valid = [], []
    for train_idx, valid_idx in folds:
        d_train = d.select(train_idx)
        d_valid = d.select(valid_idx)
        train.append(d_train)
        valid.append(d_valid)
    return train, valid

# these samples were found to be mislabeled with 100% confidence
correction = {
    "44b8ee14177f93d348d276e3416a4b57a4d2e5ca45a6e0be4a51b68dadb04940": "model_b",
    "4a312da4317035ad089e8f0c4585c2b27e1e5c89431f977468606190e94194d8": "model_b",
    "9957d12658eb588e26f816c522b3b8fd2027ad708561476460fa25e82895caad": "model_b",
    "9c5870db10aed5f68f345fd8c10902b11e99d3c33c521fe4c82aed7990213b62": "model_b",
    "389e8fde1e2bc4ba00d60f28539be0d0beb31a916263d88b145b920ff5daf564": "model_b",
    "dbbcde950433cae59ed8e022046a7bdb315fdd7a1f8633eee22d37f7b049c269": "model_a",
    "9957d12658eb588e26f816c522b3b8fd2027ad708561476460fa25e82895caad": "model_b",
    "9c5870db10aed5f68f345fd8c10902b11e99d3c33c521fe4c82aed7990213b62": "model_b",
    "2d7ed95cc00879ec044544d32698237fcef9af9999021574729e35ac59758ba4": "model_b",
    "389e8fde1e2bc4ba00d60f28539be0d0beb31a916263d88b145b920ff5daf564": "model_b",
    "dbbcde950433cae59ed8e022046a7bdb315fdd7a1f8633eee22d37f7b049c269": "model_a",
    "2fb9c6d3ef60ed3b3344eca6f409fd3023e9eb763d29a4d1290a1947fcfc9505": "model_b",
    "cca5cd1b987f1be1f9039aba7e362796f10a6b839c4515698fe686f36c7fc6a3": "model_a",
    "2920ea99d5a43d9b1ec20735ce55a12040b701da97c3f3447036bf3ed85c5b5a": "model_b",
    "c2a0a619f9c427f4fa482e85f006a64252203ac89f6ad239bf2b5ddc594ff573": "model_b",
    "e3553b95df518aa1b88a6c9b90d42cdef8e070112c2acf498a23ad293d645bb1": "model_a",
    "ff0bcae6d101a51b51766df4e314f0c6d7c374fa64b90fefd10dfe8d84e38122": "model_a",
    "db4dbd7fc596558ebf3cf70915077aedfdbc88134538dddcb0404ea2187d320b": "model_a",
    "f30981aea488189cf11525e9326b605ffe070054eadf7ba97386465bbc56931f": "model_b",
    "9c19491221c9c97bbb38ff8039d7e58d76938100911534b1eb9e79b44b6f043f": "model_a",
    "4fc2fd9dffda0c0a5ee242e4d7c3ee5dbd75c942d181c895b0d59f595553d7a5": "model_a",
    "7dd22f185056f044c0b0b4e45ae3d5cf2384057fca522148381b5484f01381d5": "model_a",
    "248663e17fc2e4983c439b8ee226ddb5db987db456f8388fadeadc1f3e909255": "model_b",
    "afa617528b368d3f9998a7338c323667bb1f780e8042589d6c24288c055ccb37": "model_b",
    "ebfedc7ddeb1ace94e76b63f2a2ed5c1d3316a71f2044aacb522d8e3e9ff2408": "model_b",
    "d3848369d74debb48cd0c0c407085d1333eb60f1bf383fcb8c95c7298a802a9f": "model_b",
    "e6da6fadc7bc6cb3350651825ab28f988f918ef6b1ecf22c71f7bc3694684aaf": "model_b",
    "77992fbf25e0933d580426f9143e00300cd8f7a4c93a4327bd694c6a3ef1b1f8": "model_a",
    "8fc3d94a238cb6a48e5e29bbaea877c42e3429d7efa00c2c85926b463bbf442e": "model_b",
    "af03f741c47339c51368a735d4d9fb587ec44879254600cbb088a6ad730d8928": "model_a",
    "d07ba742ff73371f0e7a2bb1df3e157da3ee00b57f517cfad5d283440a64e8be": "model_b",
    "c540ed69655a4ccb2669bb003acd6a9302705794eac59c84b483ae93317516ae": "model_a",
    "c2b58d328c1384ce85d22c147540a8635ed426ecfb5978b3c850e65aaf698bc5": "model_a",
    "e42820402532d862363acac323abcaf7ec0f7522f754645f1134623e2b677ac3": "model_b",
    "96ae265c82dcaabeefaaa6463884bec89f8bade3e938f660ef9c5ca72dd9d9c7": "model_b",
    "5af2ff83f068862d7851269648f2c5723adb85628ff27db7ce902b2b702945d6": "model_b",
    "5ed22e1ed3fc06e6efb0c8e17bf11f7ad20410eeb4bb5498c10951ff1b6381b8": "model_b",
    "1cc36685e641ec54cc04a9a6e4acc0a480b1235624e349d020f3be2defeb5252": "model_a",
    "3e6e311d06c84fb79941925d5cc635f0fa7cb87dd7178bf6d4e37ddb747a4312": "model_b",
    "5d85c4604f383c9f96c314abfecb937292d88eaea78645965382ad05f4e8849b": "model_a",
    "e0c4ad75e2a50c2cf9923096fb1aee6de64d5a8d2ebacb37dd9e9d0f522ca80c": "model_a",
    "da2eb6c0f1983fc0ed2acfaa2d4bc2504b5e3fc7c184b724bf5f8bf5b3ee69df": "model_b",
    "d6bc48f9d1cbb5668e0d82314868dbca1b7145fddc068cfe3aaf7ea9c4974f89": "model_a",
    "648ac299a9d96f2e8a0d6b1b63170c009b9f64930d7874651a473406ca27473d": "model_a",
    "163b0157ef07ab45e1650849da7cf775ee2428012c708711dbd0b968f1af6ac1": "model_b",
    "7142315870e5c3cca1de070605d338f8f427ad8dae10df424023258f442f86c2": "model_b",
    "12f93e97cd5196400cb158cfa2624ee33bb892f3c5fbdf215e52a7ad200d0837": "model_b",
    "c2f90bf6e12074c04a8d959b595cc02ed3ba9d7e77290714ae4acbb590f75344": "model_b",
    "fc230e27fd45d01bb2dfb90a941eeca2882535f432eaf814938eb3f8959f9a05": "model_a",
    "74252d6c65da63eec5e635640419da86714888da732c9bf64b657ae357da58d8": "model_b",
    "ba7f711481dd1b2f9b6ff1f0e9ef08665bab2d1d0446e6255b7c7e95cb205cad": "model_b",
    "baf96513be60c412c808eafe3e4dd620d1924f80678f7bbbfc7fb1c625dc94ae": "model_a",
    "d88f23c84a2a106693754decdded3527f13aba1d20b9c959423ef0ef1bdf2239": "model_a",
    "73c377ae7fe649214febcb11cbd7897d970468fa60e4cffab0741b29b2f2f8f2": "model_a",
    "b4341384cbca9ef7d37faa88624626518add20222dc174896e6293cad2a0197f": "model_b",
    "285f84972e914a3b456738e57bc145165e9aa068323ce4765d8d128122be7e50": "model_a",
    "50663a3db799d036a1bbd552f608f17a778b39d4fda711b83c32ceb162201bfd": "model_b",
    "8b2cfc11d434b77cdd737f764be4702c50948fafffafb535a91da50eeb345637": "model_a",
    "2ef00c4131dddffec6fa3c8663207911a137f29f74904273ef5f5a6600348164": "model_a",
    "a8c2f4faea302f941cabbfb41d86687cdaaa1b5a91f48c771c53f9b6a540a4b4": "model_b",
    "5f04c5152423111a8cc74adcf2548beff01be3e30910a789f97dd0c65849a4db": "model_a",
    "1b670438dd461c554aee17874f316f9b1c8bb61d6560ac8edcb7c54590db6a82": "model_a",
    "d5bac1fd4de6fd269f2862f7648f922e03da73f4caf76707f7e78cd60f0d4cd3": "model_b",
    "5c5f3a1acb0bdaace3cf8f1010dda459a38d6a7a2de3cc16beeee536371ba2b4": "model_a",
    "77587ee2452ccaea97909f7e6eb25330ecf3489640717d34feb94e7643b71dbe": "model_b",
    "417e927cd33a328eac585b81b098d2ce22764da5ebac6f43c570966f9e8cc6e8": "model_a",
    "62b23ceb7483ca59f77abdc4149a978d81679bf63526836262e4c672930b201d": "model_b",
    "2359228e6218c7a01e087b934b0d9ec9fecf7e5b58b5b2a7d3bbbba3872632c9": "model_a",
    "2fd21ce91fb242dfa16c9a2f4225505399059b46fc663ceae19538a1c97962c2": "model_b",
    "e2522fa40c11e4518f5f9a73bd5a5c74867c1e1df4203a161de8fe15d00187e0": "model_b",
    "0c53b8f02343c7a649df7b4376e3db3cd3f7401b048a3dac87adb314beaa5c0a": "model_a",
    "ec048925303d63417dd36105125ec4f82afbcfd2c02f215171383d743c5a6d6b": "model_a",
    "4a9fa4314b4a02911c4aaca4e239fc22b82dec8ce75939cc630f0a3d794230a5": "model_a",
    "4031a30c61cfc1fb27b7f80f153a74d986969efd3342ffabb870ae60b8dab533": "model_b",
    "972f1bb02597dddb061fc80b5e8f99b9703edfc1e2e925b55fb692bd8c0fb13b": "model_b",
    "bd7053d6660b4c45a6743c3e1c6ea49146cdd8e1e930a7d6c79c60ce53c898f3": "model_a",
    "e96d0333d36cba9831485f17eed8a7970c70d1c442eb2baec1eb1660fbf08647": "model_a",
    "88672d5654cbbdd27a56a1211f036381484356af71e0397023c5140d4b10c7e0": "model_a",
    "722b1deebe14b53c004a107a8c11ae41356078535e0cbaf293e5f763c7a33454": "model_a",
    "b8137d6b2f410159bb949e31960e4b52be1310e33da7a9de1a1b5ff445079e3b": "model_b",
    "fd8e6988a27566190297ff46b25f0564421420a7a8e0ceadd494fc712f1f1c8d": "model_a",
    "8815cdcbf2e53d27b037c9978a4c9506c7477852163a9d6fc18d3321dc10670a": "model_b",
    "c0c32f4877509076a0b89756890242bfdc21249bf9dea8fe87b2912f68c53dac": "model_b",
    "57d4892093e755bc836c3dd395e3467b551174f9aba5b6d81a193b6309d06f53": "model_b",
    "8d22efa9af57a65c5333f608b339b4f21846f002dddc1ae996e7629772b0dc9b": "model_a",
    "3194c36ac32d39384e475413a40cb4fc4feb4cadc5a64dd00c48dd4d90ac49d2": "model_a",
    "e5e6d42421469b331c0792bf5f12de922d009610685e1ad4d779e50a572b9a9e": "model_b",
    "26a9b9299980a1e54711764acacb4acddd76a4cf97f211803c4b87f88d4035ca": "model_b",
    "a20f5082153e036ab9d5c29757d8c9fb5d34fdfe5648b5e248e12a39231fcb7e": "model_b",
    "f286fe26e1105fd53cb550c1277c069e560505d700ad411ffedd84841a7b977f": "model_b",
    "50888ac47ed03d18434f2c188796db22cf204de82e459c2cc22c21afd563ea61": "model_b",
    "bb1a941f6614a7a8fca2c6b12e7749848089f912b91b5357c5415f326205f7ca": "model_b",
    "911d9adde03800ce37e6430e9945e84f089c4e01d8bf3d89c39cedddbcb29b06": "model_a",
    "23a8b4a93c385f11b102b2eafe8973022af28d67fee259947d69fd6c1b36d0e0": "model_a",
    "6d667c945faf168e090d435aa850d056af5c5181810468359869e9441f241a05": "model_a",
    "64e9839a3d74083f3f4af20a4425ee345cadeaced138a1e427d2e442ead676ad": "model_b",
    "22e5c643420c0cec05ff6bfbaeaef6ae49d3d2da6254a86775f2a8af11341850": "model_a",
    "508d1a0c0812fb3e9fe2c271fbcf7cf011f9df79ed81d53fa8dbd789fdc87554": "model_b",
    "61131eb40b4fa82fba6f3facc8d8e4ca991daca7904e605fa98379891232883e": "model_a",
    "bf868efa4ba4a47e42dc7010d10639eed9908cf2814b6c7cc19599ac8f0da085": "model_a",
    "5872d10781e2f0df14d9d61edc756e087a5fc1f85553781ff79a64e2eca048c7": "model_b",
    "764825844215b77ff4c73af568547f97d58c34ed935b9353e3c5e94a0da9089a": "model_b",
    "7f573762d28ea01f07de8db018258da294da299c1ab3e4f9aa8c454672f8376a": "model_a",
    "03b718ecc12d453781b4ee938220e3ea7b9109d8e0634f60a6cf0cb9fba344bf": "model_a",
    "2ac4d0256d905fcc6578c7d7100b2ceba967a12dd014016874ad12dc15cdee05": "model_a",
    "fc65b21014b943e7f1a88872c6154dce276fd8d2ebb6ca68d61479351fddeb9c": "model_b",
    "8c1325a27a96abf80bea601415c83c6a0448cebbc91b9ee579fb065347580094": "model_b",
    "fdce2b171339061172ff70be8d8537a36c6ae1e75b61f008778b75643d8243cf": "model_b",
    "f483cd8a469e978cdf32a181b065c8ef15e6dd1fcb99b8a55e9b481cc65cd31e": "model_b",
    "b3eac7f73f733b39389fd7e9f5240436b148db0232b6f29c5d4e97ddb896b5f1": "model_b",
    "790baf7546a2820f2648b9307b02b906569b9e98774d186e8b1fb2efcba24d46": "model_b",
    "eaa9c832e7690b8e0efb752dbfd0c1616cfac4711ce98c93062d89c33ec1517f": "model_a",
    "2488bcf3e907df4ac293cc17b66931e0a560bb8ada39050337dbfcef6c5fdb19": "model_b",
    "b181820d427ff34d0ed416c8ee0e776e06155be30a26e30cdbe857698010e82a": "model_b",
    "64ace6476d94ddbdc303566df824a0350a7ebee406653cd48838806228d28a99": "model_a",
    "5383e300af99febfdf83f32662d507f8bae7d1a9491b2df8775305fe60fd31a3": "model_b",
    "32de30addc5dacb11324adc30310c07dc301757390b9991f33a84aa917db5a63": "model_b",
    "104152b7f96bab9cd3b4e54737807e408f6202e41c52b397eada91bf2602510b": "model_a",
    "1d84f897b96d50da0a6ed91be698750d4ef0e74094bdcaa35e0f96bef113c102": "model_b",
    "868662ee8548b5ad61e1994276dc5558d176884f74eed5d4e39273e06a81ef6d": "model_a",
    "89d2f2d61b1121b6852266eb4e0e9651ec6407c8d903a5ba95fa2f4a8b4ebbb7": "model_a",
    "97a59186a0625dc0232a8eb1bcbdd197e117716e199baaa4e5105b8bc8b5fb13": "model_b",
    "01bf359bb888b04dd84967864b471ad6b968b6a5ea70c68857b12737027509d3": "model_a",
    "c150a76a1db472bd6d81bdc1dc2075df2bfb035f201975b334de24a0292c4958": "model_b",
    "19749c0c8cd69e66cbf458637706d7ebc9a7461e394f0cf6e9bf324d5f238f42": "model_b",
}

def correct_winner(x):
    if x['id'] in correction:
        x['winner'] = correction[x['id']]
    return x
# ;; new

dnew = load_dataset("parquet", data_files="data/train.parquet", split="train")
percent_english = sum(x['language'] == 'English' for x in dnew) / len(dnew)
print(f"English: {percent_english*100:.0f}%")

dnew = dnew.map(add_length)
dnew = dnew.remove_columns(["language"])
dnew = dnew.add_column("source", ["new"] * len(dnew))
dnew = dnew.add_column("n_turns", [1] * len(dnew))

dnew = dnew.map(hash_prompt_responses)
dnew = dnew.map(correct_winner)

dnew_train, dnew_valid = split_5_fold(dnew)
dnew_train = [remove_response_duplicates(x) for x in dnew_train]
dnew_train = [remove_duplicates(x) for x in dnew_train]
dnew_train = [balance_ab_winners_with_swap(x) for x in dnew_train]

for i, d in enumerate(dnew_valid):
    d.to_parquet(f"data/new_valid_{i}fold.parquet")

for i, d in enumerate(dnew_train):
    d.to_parquet(f"data/new_train_{i}fold.parquet")

dnew = remove_response_duplicates(dnew)
dnew = remove_duplicates(dnew)
dnew = balance_ab_winners_with_swap(dnew)
dnew.to_parquet("data/new.parquet")

hash_0valid = hashlib.sha256("".join(dnew_valid[0]['id']).encode()).hexdigest()
hash_0train = hashlib.sha256("".join(dnew_train[0]['id']).encode()).hexdigest()
hash_new = hashlib.sha256("".join(dnew['id']).encode()).hexdigest()

print(f"hash_0valid: {hash_0valid}")
print(f"hash_0train: {hash_0train}")
print(f"hash_new: {hash_new}")

# ;; old may-55k

def format_to_single(x):
    n_turns = len(eval(x["prompt"], {"null": "[null]"}))
    return {
        "prompt": eval(x["prompt"], {"null": "[null]"})[0].replace(r'\/', '/').encode('utf-8', errors='replace').decode('utf-8'),
        "response_a": eval(x["response_a"], {"null": "[null]"})[0].replace(r'\/', '/').encode('utf-8', errors='replace').decode('utf-8'),
        "response_b": eval(x["response_b"], {"null": "[null]"})[0].replace(r'\/', '/').encode('utf-8', errors='replace').decode('utf-8'),
        "winner": 'model_a' if x["winner_model_a"] else "model_b" if x["winner_model_b"] else "tie",
        "model_a": x["model_a"],
        "model_b": x["model_b"],
        "n_turns": n_turns,
    }

dold = load_dataset("lmsys/lmsys-arena-human-preference-55k", split="train")
dold = dold.map(format_to_single)
dold = dold.map(add_length)
dold = dold.remove_columns(["winner_model_a", "winner_model_b", "winner_tie", "id"])
dold = dold.add_column('source', ['old'] * len(dold))
dold = dold.map(hash_prompt_responses)
dold = remove_response_duplicates(dold)
dold = remove_duplicates(dold)
dold = dold.map(correct_winner)

# ;; very old jul-33k

def format_to_single(x):
    return {
        "prompt": x['conversation_a'][0]['content'],
        "response_a": x['conversation_a'][1]['content'],
        "response_b": x['conversation_b'][1]['content'],
        "n_turns": len(x['conversation_a']) // 2,
    }

dveryold = load_dataset("lmsys/chatbot_arena_conversations", split="train")
dveryold = dveryold.map(format_to_single)
dveryold = dveryold.remove_columns(["conversation_a", "conversation_b", "turn", "toxic_chat_tag", "openai_moderation", "anony", "judge", "question_id", "tstamp", "language"])
dveryold = dveryold.add_column('source', ['very-old'] * len(dveryold))
dveryold = dveryold.map(hash_prompt_responses)
dveryold = remove_response_duplicates(dveryold)
dveryold = remove_duplicates(dveryold)
dveryold = dveryold.map(correct_winner)

# ;; mini

dmini = load_dataset("json", data_files="data/sample_gpt-4o-mini.jsonl", split="train")

def format_to_single(x):
    prompt = x['conversation_a'][0]['content']
    response_a = x['conversation_a'][1]['content']
    response_b = x['conversation_b'][1]['content']
    return {
        "prompt": prompt,
        "response_a": response_a,
        "response_b": response_b,
        "n_turns": len(x['conversation_a']) // 2,
    }

dmini = dmini.map(format_to_single)
dmini = dmini.remove_columns(["category", "conversation_a", "conversation_b", "opponent", "outcome", "turn", "question_id", "category_tag", "language"])
dmini = dmini.add_column("source", ["mini"] * len(dmini))
dmini = dmini.map(hash_prompt_responses)
dmini = remove_response_duplicates(dmini)
dmini = remove_duplicates(dmini)
dmini = dmini.map(correct_winner)

# ;; merge and remove duplicates

dlmsys = concatenate_datasets([cast(x) for x in [dnew, dold, dveryold, dmini]])
dlmsys = remove_response_duplicates(dlmsys)
dlmsys = remove_duplicates(dlmsys)
dlmsys = balance_ab_winners_with_swap(dlmsys)

dlmsys.to_parquet("data/lmsys-130k.parquet")

dlmsys_hash = hashlib.sha256("".join(dlmsys['id']).encode()).hexdigest()
print(f"hash: {dlmsys_hash}")

# ;; separate > 1 turn and ties for pseudo labeling

dlmsys.filter(lambda x: x['n_turns'] > 1 or x['winner'] not in ['model_a', 'model_b'])
dlmsys_extra = dlmsys.filter(lambda x: x['n_turns'] > 1 or x['winner'] not in ['model_a', 'model_b'] or x['source'] != 'new')
dlmsys_extra.to_parquet("data/lmsys-old-nturn-ties.parquet")

# ;; new + old

dold_win_1turn = dold.filter(lambda x: x['winner'] in ['model_a', 'model_b'] and x['n_turns'] == 1)
dnew_and_dold_win_1turn = concatenate_datasets([dnew, dold_win_1turn])
dnew_and_dold_win_1turn = remove_duplicates(dnew_and_dold_win_1turn)
dnew_and_dold_win_1turn = remove_response_duplicates(dnew_and_dold_win_1turn)
dnew_and_dold_win_1turn = balance_ab_winners_with_swap(dnew_and_dold_win_1turn)
dnew_and_dold_win_1turn.to_parquet("data/new_and_old_win_80k.parquet")

# ;; new + old + offset

doffset = load_dataset("NCSOFT/offsetbias", split="train")
def format_to_single(x):
    return {
        "prompt": x['instruction'].strip(),
        "response_a": x['output_1'].strip(),
        "response_b": x['output_2'].strip(),
        "n_turns": 1,
        "winner": 'model_a' if x['label'] == 1 else 'model_b',
    }
doffset = doffset.map(format_to_single)
doffset = doffset.remove_columns(["instruction", "output_1", "output_2", "label"])
doffset = doffset.add_column('source', ['offset'] * len(doffset))
doffset = doffset.map(hash_prompt_responses)
doffset = remove_response_duplicates(doffset)
doffset = remove_duplicates(doffset)

dnew_and_dold_and_offset = concatenate_datasets([dnew, dold_win_1turn, doffset])
dnew_and_dold_and_offset = remove_duplicates(dnew_and_dold_and_offset)
dnew_and_dold_and_offset = remove_response_duplicates(dnew_and_dold_and_offset)
dnew_and_dold_and_offset = balance_ab_winners_with_swap(dnew_and_dold_and_offset)
dnew_and_dold_and_offset.to_parquet("data/new_old_offset_90k.parquet")

# ;; PPE

from datasets import load_dataset
ppe = load_dataset("lmarena-ai/PPE-Human-Preference-V1", split="test")
def format_to_single(x):
    return {
        "prompt": x['prompt'],
        "response_a": x['response_1'],
        "response_b": x['response_2'],
        "n_turns": x["conv_metadata"]["turns"]
    }
ppe = ppe.map(format_to_single)
ppe = ppe.map(hash_prompt_responses)
ppe = ppe.map(add_length)
ppe = remove_response_duplicates(ppe)
ppe = remove_duplicates(ppe)
ppe = ppe.map(correct_winner)

ppe1 = ppe.filter(lambda x: x['n_turns'] == 1)
ppe1win = ppe1.filter(lambda x: x['winner'] in ['model_a', 'model_b'])
ppe1win = balance_ab_winners_with_swap(ppe1win)

ppe1win.to_parquet("data/ppe-10k.parquet")
ppe_hash = hashlib.sha256("".join(ppe1win['id']).encode()).hexdigest()[:4]

ppentie = ppe.filter(lambda x: x['winner'] not in ['model_a', 'model_b'] or x['n_turns'] > 1)
ppentie = ppentie.remove_columns(["winner"])
ppentie.to_parquet("data/ppe-tie-5k.parquet")
# ;;

vibe = load_dataset("lmarena-ai/Llama-3-70b-battles", split="train")
def format_to_single(x):
    return {
        "prompt": x['question'],
        "response_a": x['response_a'],
        "response_b": x['response_b'],
        "n_turns": 1
    }
vibe = vibe.map(format_to_single)
vibe = vibe.remove_columns(["question", "question_id", "is_code", "is_refusal", "num_tokens_info", "language", "tstamp"])

vibe = vibe.map(hash_prompt_responses)
vibe = vibe.map(correct_winner)
vibe = vibe.map(add_length)
vibe = remove_response_duplicates(vibe)
vibe = remove_duplicates(vibe)
vibe = balance_ab_winners_with_swap(vibe)

vibe.to_parquet("data/vibe-1k.parquet")
