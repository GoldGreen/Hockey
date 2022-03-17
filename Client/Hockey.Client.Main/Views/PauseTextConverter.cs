using Hockey.Client.Shared.Converters;
using System;
using System.Globalization;

namespace Hockey.Client.Main.Views
{
    public class PauseTextConverter : ConverterBase<PauseTextConverter>
    {
        public override object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is bool b)
            {
                return b ? "Продолжить" : "Пауза";
            }
            return null;
        }
    }
}
