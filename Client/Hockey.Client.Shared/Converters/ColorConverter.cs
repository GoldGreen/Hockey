using System;
using System.Globalization;
using System.Windows.Media;

namespace Hockey.Client.Shared.Converters
{
    public class ColorConverter : ConverterBase<ColorConverter>
    {
        public override object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            return new SolidColorBrush((Color)value);
        }
    }
}
