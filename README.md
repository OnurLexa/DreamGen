# Discord Stable Diffusion Bot

Discord için hazırlanmış bir Stable Diffusion görsel üretim botu.  
Stability AI API kullanarak metinden görsel üretir ve Discord’a gönderir.

## Özellikler
- /resim komutu ile görsel üretme
- Prompt ve negative prompt desteği
- Ayarlanabilir parametreler: steps, cfg_scale, width, height, samples, seed, model
- Stability içerik filtresi desteği
- Kullanıcı başına bekleme süresi (cooldown)
- Aynı anda çalışabilecek işlem sınırı (concurrency limit)
- SQLite veritabanına kullanım kaydı

## Gereksinimler
- Python 3.10 veya üstü
- Discord Bot Token
- Stability AI API Key

## Kurulum
1. Bu depoyu indir veya klonla.
2. Gerekli kütüphaneleri yükle:
   ```bash
   pip install -r requirements.txt
   ```
3. `.env` dosyası oluştur ve `.env.example` içindeki değişkenleri doldur.
4. Botu çalıştır:
   ```bash
   python bot.py
   ```

## Kullanım
Discord sunucuna botu ekledikten sonra `/resim` komutunu kullanabilirsin.  

Örnek:
```
/resim prompt:"uzayda yürüyen kedi" steps:30 cfg_scale:7 width:512 height:512
```

Bot, parametrelere göre görsel üretecek ve kanala gönderecektir.

## Notlar
- Stability AI’nin ücretsiz kullanım kotası vardır, fazla kullanımda ücret çıkabilir.
- Filtreye takılan içerikler bulanık veya engellenmiş dönebilir.
- Daha fazla model için Stability AI dokümantasyonuna bakabilirsiniz.
