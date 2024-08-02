from telegram import Update,ReplyKeyboardMarkup,InlineKeyboardButton,InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes,CallbackQueryHandler,MessageHandler,filters
import torch
from torchvision import transforms
from PIL import Image
import os
from datetime import datetime
from dotenv import load_dotenv
import openai
import re
load_dotenv()
API_KEY= os.getenv('OPENAI_SECRET_KEY')
openai.api_key =API_KEY
token = os.getenv('TELEGRAM_TOKEN')
print(token)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
keywords=['rice','rice disease','blast','tungro','bacterial blight', 'brownspot']

def filter_input(user_input):
    for keyword in keywords:
         if re.search(keyword,user_input,re.IGNORECASE):
             return True
         return False
def generate_response(question):
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question}
        ]
        )
        return response.choices[0].message['content']
def chatbot_response(user_input):
    if filter_input(user_input):
        response = generate_response(user_input)
        return response
    else:
        return "I'm sorry, I can only answer questions related to rice diseases like blast, tungro, bacterial blight, and brown spot. Please ask a related question."

def load_image(image_path):
    my_image = Image.open(image_path).convert('RGB')
    my_image = transform(my_image).unsqueeze(0)
    return my_image
def predict(model,image_path):
    img = load_image(image_path)
    model.eval()
    with torch.no_grad():
        outputs=model(img)
        _,predicted = torch.max(outputs,1)
        softmax = torch.nn.Softmax(dim=1)
        probs = softmax(outputs)  # Calculate class probabilities
        accuracy = torch.max(probs).item()  # Max probability
    return predicted,accuracy
class_rice=['Bacterialblight-ជំងឺបាក់តេរី', 'Blast-ជំងឺប្លាស់', 'Brownspot-ជំងឺអុចត្នោត', 'Tungro-ជំងឺតុងក្រូ']
resNet50_1= torch.load('ResNet50-Rice-Disease-Detection-v1.pt')
languages = {'lang_en': 'lang_en(អង់គ្លេស)', 'lang_kh': 'lang_kh(ខ្មែរ)'}
lang=""
messages={
    'en':{
        'welcome':"🌾 Welcome to CropWiz, your personal rice crop guardian! 🌱 Ready to boost your yields and safeguard your rice? Let’s make those fields flourish! 🚜💪",
        'selected':"English language mode ON!!",
        'main_menu_call_to_action':[[InlineKeyboardButton('📷 Scan Rice Crop (Send the picture to us)',callback_data='scan_en')],[InlineKeyboardButton('👩‍🌾👨‍🌾 RICE FARMER Community (Join our Telegram Group)',callback_data='community_en')],[InlineKeyboardButton('❓ How to use the bot?',callback_data='instruction_en')]],
        'community':{
            'C0001': "🌾 Join our Rice Farmer Hub! Let's grow together by sharing insights and solving challenges. Tap here to connect with fellow farmers! 🌱",
            'C0002':"Join our channel🌾👨‍🌾"
        },
         'scan_rice_crop_menu':{
             'SRCM0001':"📷Please send your rice image to use. Make sure it is clear to get a better result",
             'SRCM0002':"⏳Image is processing.Please wait for 2 to 3 second.",
             'SRCM0003':"✅Done. Here is the result.",
             'SRCM0004':"❌Your picture is not clear enough to detect. Send me a better picture"
         },
    },
    'kh':{
        'selected':"អ្នករើសភាសាខ្មែរ",
        'welcome':"🌾 សូមស្វាគមន៍មកកាន់ CropWiz ដែលជាអ្នកមើលថែដំណាំស្រូវផ្ទាល់ខ្លួនរបស់អ្នក! 🌱 ត្រៀមខ្លួនរួចជាស្រេចក្នុងការបង្កើនទិន្នផល និងការពារស្រូវរបស់អ្នកហើយឬនៅ? សូមអោយដំណាំស្រូវទាំងនោះទទួលបានទិន្នផលខ្ពស់! 🚜💪",
         'main_menu_call_to_action':[[InlineKeyboardButton('📷ពិនិត្យជំងឺស្រូវ(ផ្ញើរូបភាព))',callback_data='scan_kh')],[InlineKeyboardButton('👩‍🌾👨‍🌾សមាគមន៍អ្នកដាំស្រូវ(អាចសួរសំណួរទៅកាន់កសិករដ៏ទៃទៀត)',callback_data='community_kh')],[InlineKeyboardButton('❓របៀបប្រើប្រាស់',callback_data='instruction_kh')]],
         'scan_rice_crop_menu':{
             'SRCM0001':"📷សូមថតរូបភាពស្រូវរបស់អ្នក​រួចផ្ញើមកកាន់យើងខ្ញំុ(រូបភាពត្រូវតែច្បាស់)",
             'SRCM0002':"⏳ធ្វើការពិនិត្យសូមរង់ចាំ២ ទៅ៣​ វិនាទី.............",
             'SRCM0003':"✅ពិនិត្យរួចរាល់ នេះជាលទ្ធផល",
             'SRCM0004':"❌សូមផ្ញើរូបភាពអោយច្បាស់ជាងនេះបន្តិច"
         },
         'community':{
             'C0001':'ចូលឆានែលតេលេក្រាមខាងក្រោម នេះដើម្បីចែករំលែកបទពិសោធន៍ និង ដោះស្រាយបញ្ហាផ្សេងៗទាក់ទននឹងដំណាំស្រូវ​ ជាមួយកសិករខ្មែរយើងផ្សេងទៀត',
             'C0002':"ចុចចូល🌾👨‍🌾"
         }
      
          
    }
}
user_language = {}
async def start(update:Update,context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    # if user_id in user_language:
    #     lang = user_language[user_id]
    #     update.message.reply_text(messages[lang]['selected'])
    #     print(lang)
    # else:
    buttons = [[InlineKeyboardButton(languages[lang], callback_data=lang) for lang in languages]]
    await update.message.reply_text('សូមជ្រើសរើសភាសា-Choose your language:', reply_markup=InlineKeyboardMarkup(buttons))
async def set_language(update:Update,context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    user_id = query.from_user.id
    _, lang = query.data.split('_')
    user_language[user_id]=lang
    await query.answer()
    print(lang)
    await query.edit_message_text(text=messages[lang]['selected'])
    await context.bot.send_message(chat_id=query.message.chat_id,text=messages[lang]['welcome'],reply_markup=InlineKeyboardMarkup(messages[lang]['main_menu_call_to_action']))
# async def hello(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
#     # await update.message.r  eplyKeyboardMarkup(keyboard,resize_keyboard=True)
#     await update.message.reply_text(f'សួស្តី {update.effective_user.first_name} !\n',reply_markup=reply_markup)
async def community(update:Update,context:ContextTypes.DEFAULT_TYPE)->None:
    query = update.callback_query
    _, lang = query.data.split('_')
    await query.answer()
    print( messages[lang]['community']['C0001'])
    channel_url="https://t.me/crop_wiz_community"
    text_message = messages[lang]['community']['C0001']
    button = InlineKeyboardButton(messages[lang]['community']['C0002'],url=channel_url)
    markup = InlineKeyboardMarkup([[button]])
    # await query.edit_message_text(text=text_message, reply_markup=markup)
    await context.bot.send_message(chat_id=query.message.chat_id,text=text_message,reply_markup=markup)
async def scan(update:Update,context:ContextTypes.DEFAULT_TYPE)->None:
    query = update.callback_query
    _, lang = query.data.split("_")
    await query.answer()
    text_message= messages[lang]['scan_rice_crop_menu']['SRCM0001']
    await context.bot.send_message(chat_id=query.message.chat_id,text=text_message,)
async def handle_reply(update:Update,context:ContextTypes.DEFAULT_TYPE)->None:
    #  query = update.callback_query
    # #  _, lang = query.data.split("_")
    #  await query.answer()
     user_message = update.message.text
     user_message=user_message+" generate 5 bullet point keep it short and conscise."
     result= chatbot_response(user_message)
     await update.message.reply_text(result)
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.photo:
        await update.message.reply_text("Please send a photo.")
        return  # Exit the function because there is no photo to process
    lang='en'
    images_dir = 'images'
    os.makedirs(images_dir, exist_ok=True)
    image_path = 'received_photo.jpg'
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    image_path = os.path.join(images_dir, f'image_{current_time}.jpg')
    photo = update.message.photo[-1]
    new_file = await context.bot.get_file(photo.file_id)
    await new_file.download_to_drive(image_path)
    await update.message.reply_text(f"{messages['kh']['scan_rice_crop_menu']['SRCM0002']}\n\n{messages['en']['scan_rice_crop_menu']['SRCM0002']}")
    class_index, accuracy = predict(resNet50_1,image_path)  # Assuming predict is an async function
    print(accuracy)
    if accuracy > 0.70:
        current_time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        format_accuracy = 100*accuracy
        format_accuracy =  "%.2f" % format_accuracy
        result_message = f'ថ្ងៃបរិច្ឆេទ/Date :{current_time}\n\n'+ f'លទ្ធផល(Result): {class_rice[class_index]}'+f'\n\nភាពត្រឹមត្រូវ(Accuracy): {format_accuracy}'
        await update.message.reply_text(result_message)
    else:
         await update.message.reply_text(f"{messages['kh']['scan_rice_crop_menu']['SRCM0004']}\n\n{messages['en']['scan_rice_crop_menu']['SRCM0002']}")

def main():
    TOKEN=token
    # app = ApplicationBuilder().token(TOKEN).build()
    # app.add_handler(CommandHandler("start", start))
    # app.add_handler(CallbackQueryHandler(set_language,pattern='^lang_'))
    # app.add_handler(CallbackQueryHandler(community,pattern='^community_[a-z]{2}$'))
    # app.add_handler(CallbackQueryHandler(scan,pattern='^scan_[a-z]{2}$'))
    # app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    try:
    # Main bot code here
        app = ApplicationBuilder().token(TOKEN).build()
        app.add_handler(CommandHandler("start", start))
        app.add_handler(CallbackQueryHandler(set_language,pattern='^lang_'))
        app.add_handler(CallbackQueryHandler(community,pattern='^community_[a-z]{2}$'))
        app.add_handler(CallbackQueryHandler(scan,pattern='^scan_[a-z]{2}$'))
        app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
        app.add_handler(MessageHandler(filters.TEXT,handle_reply))
        app.run_polling()
    except Exception as e:
       print(f"Exception occurred: {str(e)}") 
    # app.add_handler(MessageHandler(filters.TEXT,handle_reply))
if __name__=='__main__':
    main()