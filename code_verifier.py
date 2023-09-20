import secrets
code = "def5020074e80cc29677bc6c5d214fb804df0672bb4770babf05cb29cf118a1321e997da293d7f904b28d62556bde3f8f45b95805ad8cfff1a1edd04ee4dc1593d41341378b95aa378ba83b52c259a8d49c07467ad2bfa528942db37677135b32ebeb611659b3b3c79023fdcfa79eacaed5fedd8112beb83836fa22579902b648186c3f42be422840fe66badf127583c09d57cf62322d685fdf9cce04e10426bb7ea5a72a627d14efac40e14ae0a735bb4dd8f34f2c3654341e4fe4cf2bd9d2103e983536724244fd943bcea889d310a871973efc5a72bef0d140b7c50167ece78a3de860d5fe986886f1ac2eb3657be77beac96dcfb81dfd6b528d07399948c2d84b9883f0b2bed2fd025ec74ff86078a91f126f4b842dffc86903115f510939f624743125d5eb1265042fcf2e7b2eb424b51f58776c734331d91f23eb80a50d145a73e17f323050d75b9d479cc245e281bb0b54b25d5775ea18299c2d09e2e1f82a107ccdfa455b6e1bee99087d1a88fae0cd200a7b1a13532adea115fe7a32f323899b4504a244504b5867a6abfe8fbc9405af9d34ecc9f2831d3e7b2fee42fdd80593ea7723d7f157e9c6ac024af9e9febc93bf0418a0b0ab96ee04ac0718a4d380b4aade6c78a5900f941b7ea7ea46b90d383ed778a81e649f06d38277bc5cb48f478a3957ea6"
def get_new_code_verifier() -> str:
    token = secrets.token_urlsafe(100)
    return token[:128]

code_verifier = code_challenge = get_new_code_verifier()

print(len(code_verifier))
print(code_verifier)


